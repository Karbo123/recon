import torch

logger = None
logging_group = None # the group specialized for logging
message_buffer = list()


def set_logger(new_logger):
    global logger
    logger = new_logger # modify inplace


def logger_info(message=None, header="[rank-{rank}] ", collective=False):
    """ log the message via a logger

    Args:
        message (str/None): the message to log
        header (str): the header string of the message
        collective (bool): whether need to synchronize messages from all processes (only used if multi-gpus training)
    Note:
        every process must call this function, not allowed in a diverged condition statement
        if multi-gpus, it must be used after distributed training is initialized (e.g. used in config file is okey)
    """
    assert message is None or isinstance(message, str)
    global message_buffer, logging_group

    if not torch.distributed.is_initialized(): # single gpu
        if isinstance(message, str):
            message_in = header.format(rank=0) + message
            message_buffer += [message_in]
        if logger is None: return
        # log message
        for msg in message_buffer:
            logger.info(msg)
        message_buffer = list() # clear all
    
    else: # multi-gpus
        if logging_group is None:
            logging_group = torch.distributed.new_group(backend="gloo")
        rank = torch.distributed.get_rank(logging_group)
        num_rank = torch.distributed.get_world_size(logging_group)

        if isinstance(message, str):
            message_in = header.format(rank=rank) + message
        else: message_in = None

        if not collective: # print immediately for rank-0 process
            assert rank == 0, f"rank-{rank} process (rank > 0) need to print with `collective=True`"
            if isinstance(message_in, str):
                message_buffer += [message_in]
            if logger is None: return
            # log message
            for msg in message_buffer:
                logger.info(msg)
            message_buffer = list() # clear all

        else: # need to synchronize messages
            # collection
            collected_messages = [None for _ in range(num_rank)] if rank == 0 else None

            # NOTE reach here simultaneously
            torch.distributed.gather_object(message_in, collected_messages, group=logging_group)

            if rank == 0:
                # prepare message
                message_buffer += [msg for msg in collected_messages if isinstance(msg, str)]

                # save to buffer only
                if logger is None: return # has been saved to `message_buffer`

                # log message
                for msg in message_buffer:
                    logger.info(msg)
                message_buffer = list() # clear all


