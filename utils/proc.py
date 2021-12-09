

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

