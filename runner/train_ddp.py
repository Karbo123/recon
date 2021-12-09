""" an universal runner for training model in either `distributed data parallel` mode or `single gpu` mode
"""

import os
import re
import torch
import argparse
from gorilla.core.launch import _find_free_port


def worker(rank, args):
    (args, args_unknown), (dist_config, delayed_messages) = args

    ##########################################################################
    ############################ Loading Packages ############################
    ##########################################################################

    import time
    import gorilla
    from glob import glob
    from os.path import join, abspath, basename, splitext, relpath, exists as file_exists, exists as dir_exists
    from recon.utils import parse_unknown_args, backup_config, backup_cmdinput, get_git_hash, print_cfg, set_logger, logger_info

    ##########################################################################
    ########################## initialize process ############################
    ##########################################################################

    num_rank = dist_config["WORLD_SIZE"]
    os.environ["RANK"] = str(rank)
    for k, v in dist_config.items(): os.environ[k] = str(v)
    if num_rank > 1:
        torch.distributed.init_process_group("nccl") # NOTE `nccl` can only communicate cuda memory data
        host_group = torch.distributed.new_group(backend="gloo") # NOTE therefore we use `gloo` to communicate host memory data
        torch.distributed.barrier() # wait all processes to be initialized
    torch.cuda.set_device(f"cuda:{rank}") # set device

    # print delayed messages
    if rank == 0:
        for msg in delayed_messages:
            logger_info(msg) # distributed training must be initialized
    del delayed_messages
    
    if num_rank > 1: logger_info(f"initialized sub-process (rank = {rank}, pid = {os.getpid()})", collective=True)
    else: logger_info(f"initialized main process (pid = {os.getpid()})")
    if num_rank > 1: logger_info(f"initialized distributed training environment", collective=True)

    ##########################################################################
    ############################## Configuration #############################
    ##########################################################################

    cfg = gorilla.Config.fromfile(args.cfg)
    cfg = gorilla.merge_cfg_and_args(cfg, args)
    cfg.merge_from_dict(parse_unknown_args(args_unknown)) # it may override those from file

    # NOTE
    # 1. if you want to resume from some specific output folder 
    #    (assuming you do not tell save.base_dir in config file),
    #    please pass: "--save.base_dir XXXX" as additional argument,
    #    otherwise it may create a new output folder
    # 2. please alway set the environment variable `CUDA_VISIBLE_DEVICES`
    #    - use single gpu if `CUDA_VISIBLE_DEVICES` specifies only one card
    #    - use multiple gpus if `CUDA_VISIBLE_DEVICES` specifies more than one card
    # 3. everything initialized at config file must be on CPU, and without DistributedDataParallel applied
    # 4. model initialized at config file may be used with `convert_sync_batchnorm`

    ##########################################################################
    ############################# Training Setting ###########################
    ##########################################################################

    epoch = 0
    iteration = 0
    epoch_end = cfg.get("training", {}).get("epoch_end", int(1e20))  # num of total epochs
    
    print_every = cfg.get("routine", {}).get("print_every", -1)
    checkpoint_latest_every = cfg.get("routine", {}).get("checkpoint_latest_every", -1)
    checkpoint_every = cfg.get("routine", {}).get("checkpoint_every", -1)
    validate_every = cfg.get("routine", {}).get("validate_every", -1)
    visualize_every = cfg.get("routine", {}).get("visualize_every", -1)

    metric_val_best = float("nan")
    if validate_every > 0:
        model_selection_metric = cfg.get("save", {}).get("model_selection_metric")
        model_selection_mode = cfg.get("save", {}).get("model_selection_mode")
        assert model_selection_mode in ["maximize", "minimize"], "`model_selection_mode` must be either `maximize` or `minimize`"
        model_selection_sign = 1 if model_selection_mode == "maximize" else (-1)
        metric_val_best = - model_selection_sign * float("+inf") # set to the worst

    moving_loss = 1e3 # the moving-average loss

    ##########################################################################
    ############################## Saving Setting ############################
    ##########################################################################

    working_dir = abspath(".")

    cfg_name = splitext(basename(cfg.cfg))[0].replace('cfg_', '')
    save_dir = join(working_dir, "out", cfg.get("save", {}).get("base_dir",  
                    f"{cfg_name}_{gorilla.timestamp()}"))
    if dir_exists(save_dir) and rank == 0:
        if args.reuse_folder == "always":
            print(f"reusing an existing folder: {save_dir}") # NOTE use `print` because it should be show immediately
        elif args.reuse_folder == "never":
            print(f"found existing folder: {save_dir}")
            print("abort"); exit(1)
        else:
            timeout_sec = int(re.search(r"\d+", args.reuse_folder).group())
            print("the save folder already exists, override? (timeout = yes)")
            print(f"waiting for {timeout_sec} sec...")
            time.sleep(timeout_sec)
    if rank == 0: os.makedirs(save_dir, exist_ok=True)
    if num_rank > 1:
        torch.distributed.barrier() # wait timeout

    checkpoint_dir = join(save_dir, "checkpoint")
    if rank == 0: os.makedirs(checkpoint_dir, exist_ok=True)

    repo_info = "None"
    if cfg.get("repo_dirs"): # some repos that really matter, we need to print them out
        repo_dirs = cfg.get("repo_dirs")
        repo_info = "\n"
        if isinstance(repo_dirs, dict):
            for repo_name, repo_dir in repo_dirs.items():
                repo_info += f"- [{repo_name}] dir = {repo_dir}, hash = {get_git_hash(repo_dir)}\n"
        elif isinstance(repo_dirs, (tuple, list)):
            for idx, repo_dir in enumerate(repo_dirs):
                repo_info += f"- [{basename(repo_dir)}] dir = {repo_dir}, hash = {get_git_hash(repo_dir)}\n"
        else:
            if rank == 0:
                logger_info(f"error: unrecognized repo_dirs of type {type(repo_dirs)}")
            exit(1)

    cmdlog_dir = join(save_dir, "cmdlog")
    if rank == 0: os.makedirs(cmdlog_dir, exist_ok=True)
    logfile_path = join(cmdlog_dir, "cmdlog.log")
    if rank == 0:
        logger = gorilla.get_logger(log_file=logfile_path, name=f"{basename(working_dir)}-{cfg_name}")
        set_logger(logger) # set the logger so that config file can appropriately print info via logger
        # print basic info
        logger_info(f"repo info = {repo_info}")
        logger_info(f"configuration = {print_cfg(cfg)}")

    if not cfg.no_tensorboard:
        tensorboard_dir = join(save_dir, "tensorboard")
        if rank == 0: os.makedirs(tensorboard_dir, exist_ok=True)

    if visualize_every > 0:
        vis_dir = join(save_dir, "vis")
        if rank == 0: os.makedirs(vis_dir, exist_ok=True)
    
    # backup codes
    folders_to_be_backup = glob(join(working_dir, "*"))
    folders_to_be_backup = [s for s in folders_to_be_backup if \
                            all([(x not in s) for x in ["config", "out"]])] # except these folders
    if rank == 0:
        gorilla.backup(
            backup_dir=join(relpath(save_dir, working_dir), "backup"),
            backup_list=[relpath(s, working_dir) for s in folders_to_be_backup],
            logger=logger,
        )
        backup_config(log_dir=save_dir, cfg=cfg)
        backup_cmdinput(log_dir=save_dir)

    ##########################################################################
    ############################ Dataset setup ###############################
    ##########################################################################

    # data loader
    train_loader = cfg.get("dataloaders", {}).get("train")
    if validate_every > 0:
        val_loader = cfg.get("dataloaders", {}).get("val")


    # the data to visualize
    if visualize_every > 0:
        vis_data = cfg.get("vis_data")
        if vis_data is None:
            if validate_every > 0:
                vis_data = next(iter(val_loader))
            else:
                vis_data = next(iter(train_loader))

    ##########################################################################
    ############################# Model setup ################################
    ##########################################################################

    model = cfg.get("model").to(rank)

    optimizer = cfg.get("optimizer")
    scheduler = cfg.get("scheduler")

    arg_scheduler_is_loss = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    # Print model
    if rank == 0:
        logger_info(f"Model = {model}")
        logger_info("\n" + gorilla.parameter_count_table(model))
    
    ##########################################################################
    ############################## Checkpoints  ##############################
    ##########################################################################

    load_model_path = join(checkpoint_dir, cfg.load_model_name)
    if rank == 0:
        if file_exists(load_model_path):
            meta = gorilla.resume(model=model, 
                                  filename=load_model_path,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  resume_optimizer=(not cfg.load_model_only),
                                  resume_scheduler=(not cfg.load_model_only),
                                ) # NOTE load model file only on the first process
            logger_info(f"we successfully load model parameters from: {load_model_path}")
            if not cfg.load_model_only:
                epoch = meta.get("epoch")
                iteration = meta.get("iteration")
                metric_val_best = meta.get("metric_val_best")
                iteration += 1 # NOTE the saved model is for this iteration, so we need to increase it by one to start training
                logger_info(f"we successfully load training meta (epoch, iteration, metric_val_best).")
        
        logger_info(f"epoch starting from {epoch}")
        logger_info(f"iteration starting from {iteration}")
        if validate_every > 0:
            logger_info(f"current best validation metric ({model_selection_metric}, "
                        f"{'higher is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                        f") is {metric_val_best:.3e}")

        if not args.no_tensorboard:
            tb_writer = gorilla.TensorBoardWriter(logdir=tensorboard_dir)
    
    elif file_exists(load_model_path) and not cfg.load_model_only:
        epoch, iteration, metric_val_best = None, None, None

    if num_rank > 1:
        pack = [epoch, iteration, metric_val_best]
        torch.distributed.broadcast_object_list(pack, group=host_group) # copy from the first process
        epoch, iteration, metric_val_best = pack
    
    # convert to DDP model
    if num_rank > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    ##########################################################################
    ################################ Training ################################
    ##########################################################################

    train_step = cfg.get("train_step_fn")
    evaluate = cfg.get("evaluate_fn")
    visualize = cfg.get("visualize_fn")
    
    if num_rank > 1:
        torch.distributed.barrier() # wait all processes to be here, and ready to start training
    timer_print = gorilla.Timer()

    while True:
        for batch in train_loader:
            loss_dict = train_step(batch)

            # compute the average of the loss, and so as to do logging
            if num_rank > 1:
                collected_loss_dict = [None for _ in range(num_rank)] if rank == 0 else None
                torch.distributed.gather_object(loss_dict, collected_loss_dict, group=host_group)
                if rank == 0:
                    averaged_loss_dict = dict()
                    for k in loss_dict.keys():
                        averaged_loss_dict[k] = sum(x[k] for x in collected_loss_dict) / num_rank
                    loss_dict = averaged_loss_dict

            # save to tensorboard
            if not args.no_tensorboard and rank == 0:
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"train/{k}", v, iteration)

            # print to cmdline
            if print_every > 0 and (iteration + 1) % print_every == 0 and rank == 0:
                cmd_str = "" if num_rank == 1 else "[AVG] "
                cmd_str += f"[Epoch={epoch}|Time={timer_print.since_last():.3f}] " # NOTE the time is only measured by the first process
                for k, v in loss_dict.items():
                    cmd_str += f"{k}={v:.4e} | " # NOTE the loss value is averaged over all processes
                cmd_str = cmd_str[:-2]
                logger_info(cmd_str)
                
            # save checkpoint
            if checkpoint_latest_every > 0 and (iteration + 1) % checkpoint_latest_every == 0 and rank == 0:
                save_checkpoint_path = join(checkpoint_dir, "model-latest.pt")
                gorilla.save_checkpoint(model=model, 
                                        filename=save_checkpoint_path,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        meta=dict(
                                            epoch=epoch,
                                            iteration=iteration,
                                            metric_val_best=metric_val_best,
                                        ),
                                    )
                logger_info(f"saved latest checkpoint to file: {save_checkpoint_path}")

            # save checkpoint
            if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0 and rank == 0:
                save_checkpoint_path = join(checkpoint_dir, f"model-e{epoch}-i{iteration}.pt")
                gorilla.save_checkpoint(model=model,
                                        filename=save_checkpoint_path,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        meta=dict(
                                            epoch=epoch,
                                            iteration=iteration,
                                            metric_val_best=metric_val_best,
                                        ),
                                    )
                logger_info(f"saved current checkpoint to file: {save_checkpoint_path}")

            # run validation
            if validate_every > 0 and (iteration + 1) % validate_every == 0:
                if rank == 0:
                    logger_info("performing evaluation...")
                eval_dict = evaluate(val_loader) # returning such as loss, metric
                # gather all
                if num_rank > 1:
                    collected_eval_dict = [None for _ in range(num_rank)] if rank == 0 else None
                    torch.distributed.gather_object(eval_dict, collected_eval_dict, group=host_group)
                if rank == 0:
                    # compute average
                    if num_rank > 1:
                        averaged_eval_dict = dict()
                        for k in eval_dict.keys():
                            averaged_eval_dict[k] = sum(x[k] for x in collected_eval_dict) / num_rank
                        eval_dict = averaged_eval_dict
                    # select our concerned metric
                    metric_val = eval_dict[model_selection_metric]
                    logger_info(
                        f"new averaged validation metric ({model_selection_metric}, "
                        f"{'higher is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                        f") is {metric_val:.3e}"
                    )
                    if not args.no_tensorboard:
                        for k, v in eval_dict.items():
                            tb_writer.add_scalar(f"val/{k}", v, iteration)

                    if model_selection_sign * (metric_val - metric_val_best) > 0:
                        metric_val_best = metric_val # update the best record
                        logger_info(
                            f"new best model! Metric ({model_selection_metric}, "
                            f"{'higher is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                            f") is {metric_val_best:.3e}"
                        )
                        save_checkpoint_path = join(checkpoint_dir, "model-best.pt")
                        gorilla.save_checkpoint(model=model, 
                                                filename=save_checkpoint_path,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                meta=dict(
                                                    epoch=epoch,
                                                    iteration=iteration,
                                                    metric_val_best=metric_val_best,
                                                ),
                                            )
                        logger_info(f"saved best checkpoint to file: {save_checkpoint_path}")

            # visualize performance
            if visualize_every > 0 and (iteration + 1) % visualize_every == 0:
                if rank == 0:
                    logger_info("performing visualization...")
                visualize(vis_dir, vis_data, logger) # NOTE each process will do their own visualization
            
            if rank == 0:
                logger.handlers[0].flush()
            iteration += 1

        epoch += 1

        if arg_scheduler_is_loss:
            loss_val = [loss_dict["loss"]] if rank == 0 else [None]
            if num_rank > 1:
                torch.distributed.broadcast_object_list(loss_val, group=host_group) # copy from the first process
            loss_val = loss_val[0]
            moving_loss = moving_loss * 0.9 + loss_val * 0.1
            arg_in_scheduler = (moving_loss, )
        else: arg_in_scheduler = tuple()
        scheduler.step(*arg_in_scheduler)
        
        if epoch_end > 0 and epoch >= epoch_end:
            break

    if rank == 0:
        logger_info("training completed.")
    if num_rank > 1:
        torch.distributed.destroy_process_group()



if __name__ == "__main__":

    ##########################################################################
    ############################### Configuration ############################
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--no_tensorboard", action="store_true", help="do not log tensorboard")
    parser.add_argument("--load_model_only", action="store_true", help="only load model parameters from file")
    parser.add_argument("--load_model_name", type=str, default="model-latest.pt", help="the model file name to load from")
    parser.add_argument("--reuse_folder", type=str, default="timeout-60sec", help="whether to reuse an existing folder (timeout = yes)")
    args, args_unknown = parser.parse_known_args()

    ##########################################################################
    ############################### GPU Setting ##############################
    ##########################################################################
    
    # check
    assert os.path.exists(args.cfg), f"config file: {args.cfg} does not exist"
    assert args.reuse_folder == "always" or \
           args.reuse_folder == "never" or \
           (re.match(r"timeout-\d+sec", args.reuse_folder) is not None), f"input an unknown reuse_folder: {args.reuse_folder}"

    # set GPUs
    assert torch.cuda.is_available(), "cannot find GPU"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("please manually set the environment variable `CUDA_VISIBLE_DEVICES` to specify gpus to run!")
        exit(1)

    delayed_messages = list()
    def delayed_print(message): # NOTE printed messages need to be log to file
        global delayed_messages
        assert isinstance(message, str)
        delayed_messages += [message]

    # compute world size
    gpu_ids_str = os.environ["CUDA_VISIBLE_DEVICES"]
    delayed_print(f"CUDA_VISIBLE_DEVICES = {gpu_ids_str}")
    world_size = len(gpu_ids_str.split(","))
    if world_size > 1: delayed_print(f"ready to use {world_size} gpus (DistributedDataParallel) to train...")
    else: delayed_print(f"ready to one single gpu to train...")

    # prepare configs
    dist_config = dict(WORLD_SIZE=world_size, 
                       MASTER_ADDR="localhost", 
                       MASTER_PORT=str(_find_free_port()) if world_size > 1 else "",
                    )
    worker_args = ((args, args_unknown), (dist_config, delayed_messages))

    # start
    if world_size > 1:
        torch.multiprocessing.spawn(worker,
                                    args=(worker_args, ),
                                    nprocs=world_size,
                                    join=True,
                                )
    else:
        worker(0, worker_args)

