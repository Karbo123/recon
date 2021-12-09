import torch
import pickle
import numpy as np


def obj_to_tensor(obj):
    """ convert a pickable object to a tensor (for sending obj via distributed communication)
    """
    data = pickle.dumps(obj)
    array = np.frombuffer(data, dtype=np.uint8)
    tensor = torch.from_numpy(array)
    return tensor


def obj_from_tensor(tensor):
    """ convert a tensor to a pickable object
    """
    array = tensor.numpy()
    buffer = array.tobytes()
    obj = pickle.loads(buffer)
    return obj

