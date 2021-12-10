""" some utils that are useful to write config file
"""


def dict_to_numerics(the_dict):
    return {k : v.item() for k, v in the_dict.items()}


def dict_save_lr(the_dict, optimizer, exit_min_main_lr=None, lr_word_format="lr{index}"):
    param_groups = optimizer.param_groups
    len_param_groups = len(param_groups)
    for i in range(len_param_groups):
        lr = param_groups[i]["lr"]
        the_dict[lr_word_format.format(index="" if len_param_groups == 1 else i)] = lr # save lr
        # if learning rate too small, exit
        if exit_min_main_lr is not None:
            if i == 0 and lr < exit_min_main_lr:
                return True # return True to tell need to exit


def dist_info():
    import os, torch
    rank = int(os.environ.get("RANK", "0"))
    num_rank = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}")
    return rank, num_rank, device


def dist_samplers(dataset_train, dataset_test):
    from torch.utils.data.distributed import DistributedSampler
    _, num_rank, _ = dist_info()
    samplers = dict(train = DistributedSampler(dataset_train) if num_rank > 1 else None,
                    val   = DistributedSampler(dataset_test ) if num_rank > 1 else None)
    return samplers


def kwargs_shuffle_sampler(samplers, mode="train"):
    assert mode in ("train", "val")
    sampler = samplers[mode]
    kwargs = dict(shuffle=(sampler is None), sampler=sampler)
    if mode == "val": kwargs["shuffle"] = False
    return kwargs

