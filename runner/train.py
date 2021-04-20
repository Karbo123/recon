""" Basic runner for training model
"""
import os
import torch
import gorilla
import argparse
from tensorboardX import SummaryWriter
from os.path import join, dirname, abspath, basename, splitexit, exists as file_exists
from recon.utils import parse_unknown_args

if __name__ == "__main__":
    ##########################################################################
    ############################### Configuration ############################
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--gpu_id", type=str, nargs="+", default="auto", help="which gpu id to use (higher priority)")
    parser.add_argument("--gpu_num", type=int, default=1, help="how many gpu to use")
    parser.add_argument("--parallel_mode", type=str, default="data", help="which parallel strategy to use")
    parser.add_argument("--no_tensorboard", action="store_true", help="Do not log tensorboard")
    parser.add_argument("--load_model_only", action="store_true", help="only load model params from file")
    parser.add_argument("--load_model_name", action=str, default="model-latest.pt", help="the model file name to load from")
    args, args_unknown = parser.parse_known_args()

    # configuration
    cfg = gorilla.Config.fromfile(args.cfg)
    cfg = gorilla.merge_cfg_and_args(cfg, args)
    cfg.merge_from_dict(parse_unknown_args(args_unknown)) # it may override those from file

    ##########################################################################
    ############################### GPU Setting ##############################
    ##########################################################################
    
    # set GPUs
    assert torch.cuda.is_available(), "cannot find GPU"
    gorilla.set_cuda_visible_devices(gpu_ids = (None if ("auto" in cfg.gpu_id) else cfg.gpu_id),
                                     num_gpu = cfg.gpu_num,
                                     mode    = "process")

    # for parallel training
    assert cfg.parallel_mode in ["data", "pipeline"], "`parallel_mode` must be either 'data' or 'pipeline'"
    is_data_parallel = (len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1)

    ##########################################################################
    ############################# Training Setting ###########################
    ##########################################################################

    epoch_end = cfg.get("training", {}).get("epoch_end", int(1e20))  # num of total epochs
    epoch = 0
    iteration = 0

    ##########################################################################
    ############################## Saving Setting ############################
    ##########################################################################

    working_dir = dirname(dirname(abspath(cfg.cfg)))

    save_dir = join(working_dir, "out", cfg.get("save", {}).get("base_dir",  
                    f"{splitexit(basename(cfg.cfg))[0].replace('cfg_', '')}_{gorilla.timestamp()}"))
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_dir = join(save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    cmdlog_dir = join(save_dir, "cmdlog")
    os.makedirs(cmdlog_dir, exist_ok=True)

    if not cfg.no_tensorboard:
        tensorboard_dir = join(save_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)

    if visualize_every > 0:
        vis_dir = join(save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
    

    model_selection_metric = cfg.get("save", {}).get("model_selection_metric")
    model_selection_mode = cfg.get("save", {}).get("model_selection_mode")
    assert model_selection_mode in ["maximize", "minimize"], "`model_selection_mode` must be either `maximize` or `minimize`"
    model_selection_sign = 1 if model_selection_mode == "maximize" else (-1)
    metric_val_best = - model_selection_sign * float("+inf") # set to the worst

    print_every = cfg.get("routine", {}).get("print_every", -1)
    checkpoint_latest_every = cfg.get("routine", {}).get("checkpoint_latest_every", -1)
    checkpoint_every = cfg.get("routine", {}).get("checkpoint_every", -1)
    validate_every = cfg.get("routine", {}).get("validate_every", -1)
    visualize_every = cfg.get("routine", {}).get("visualize_every", -1)

    # save log file
    logfile_path = join(cmdlog_dir, "cmdlog.log")
    logger = gorilla.get_logger(log_file=logfile_path, name=basename(working_dir))
    logger.info(f"This process pid = {os.getpid()}")
    logger.info(f"Configuration = {cfg}")

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

    model = cfg.get("model")

    if is_data_parallel:
        model = torch.nn.DataParallel(model) # TODO use DistributedDataParallel for faster speed
    model.cuda()

    optimizer = cfg.get("optimizer")
    schedular = cfg.get("schedular")

    # Print model
    logger.info(f"Model = {model}")
    n_parameters = sum(p.numel() for p in model.parameters())
    n_parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters of this model = {n_parameters}")
    logger.info(f"Total number of trainable parameters of this model = {n_parameters_trainable}")

    ##########################################################################
    ############################## Checkpoints  ##############################
    ##########################################################################

    load_model_path = join(checkpoint_dir, cfg.load_model_name)
    if file_exists(load_model_path):
        meta = gorilla.resume(model=model, 
                              filename=load_model_path,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              resume_optimizer=(not cfg.load_model_only),
                              resume_scheduler=(not cfg.load_model_only),
                            )
        if not cfg.load_model_only:
            epoch = meta.get("epoch")
            iteration = meta.get("iteration")
            metric_val_best = meta.get("metric_val_best")
    
    logger.info(f"Epoch starting from {epoch}")
    logger.info(f"Iteration starting from {iteration}")
    logger.info(f"Current best validation metric ({model_selection_metric}, "
                f"{'high is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                f") is {metric_val_best:.3e}")

    if not args.no_tensorboard:
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    ##########################################################################
    ################################ Training ################################
    ##########################################################################

    train_step = cfg.get("train_step_fn")
    evaluate = cfg.get("evaluate_fn")
    visualize = cfg.get("visualize_fn")

    timer_print = gorilla.Timer()

    while True:
        for batch in train_loader:
            loss_dict = train_step(batch)

            # save to tensorboard
            if not args.no_tensorboard:
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"train/{k}", v, iteration)

            # print to cmdline
            if print_every > 0 and (iteration + 1) % print_every == 0:
                cmd_str = f"[Epoch={epoch}|Time={timer_print.since_last():.3f}] "
                for k, v in loss_dict.items():
                    cmd_str += f"{k}={v:.4e} |"
                cmd_str = cmd_str[:-1]
                logger.info(cmd_str)
                
            # save checkpoint
            if checkpoint_latest_every > 0 and (iteration + 1) % checkpoint_latest_every == 0:
                save_checkpoint_path = join(checkpoint_dir, "model-latest.pt")
                gorilla.save_checkpoint(model=model, 
                                        filename=save_checkpoint_path,
                                        optimizer=optimizer,
                                        scheduler=scheduler
                                        meta=dict(
                                            epoch=epoch,
                                            iteration=iteration,
                                            metric_val_best=metric_val_best,
                                        ),
                                    )
                logger.info(f"Saved latest checkpoint to file: {save_checkpoint_path}")

            if checkpoint_every > 0 and (iteration + 1) % checkpoint_every == 0:
                save_checkpoint_path = join(checkpoint_dir, f"model-e{epoch}-i{iteration}.pt")
                gorilla.save_checkpoint(model=model,
                                        filename=save_checkpoint_path,
                                        optimizer=optimizer,
                                        scheduler=scheduler
                                        meta=dict(
                                            epoch=epoch,
                                            iteration=iteration,
                                            metric_val_best=metric_val_best,
                                        ),
                                    )
                logger.info(f"Saved current checkpoint to file: {save_checkpoint_path}")

            # run validation
            if validate_every > 0 and (iteration + 1) % validate_every == 0:
                logger.info("Performing evaluation...")
                eval_dict = evaluate(val_loader) # returning such as loss, metric
                metric_val = eval_dict[model_selection_metric]
                logger.info(
                    f"New validation metric ({model_selection_metric}, "
                    f"{'high is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                    f") is {metric_val:.3e}"
                )
                if not args.no_tensorboard:
                    for k, v in eval_dict.items():
                        tb_writer.add_scalar(f"val/{k}", v, iteration)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val # update the best record
                    logger.info(
                        f"New best model! Metric ({model_selection_metric}, "
                        f"{'high is better' if model_selection_mode == 'maximize' else 'lower is better'}"
                        f") is {metric_val_best:.3e}"
                    )
                    save_checkpoint_path = join(checkpoint_dir, "model-best.pt")
                    gorilla.save_checkpoint(model=model, 
                                            filename=save_checkpoint_path,
                                            optimizer=optimizer,
                                            scheduler=scheduler
                                            meta=dict(
                                                epoch=epoch,
                                                iteration=iteration,
                                                metric_val_best=metric_val_best,
                                            ),
                                        )
                    logger.info(f"Saved best checkpoint to file: {save_checkpoint_path}")

            # visualize performance
            if visualize_every > 0 and (iteration + 1) % visualize_every == 0:
                logger.info("Performing visualization...")
                visualize(vis_dir, vis_data)

            iteration += 1

        epoch += 1

        schedular.step()
        if epoch_end > 0 and epoch >= epoch_end:
            break

    logger.info("Training completed.")

