import torch
import hashlib
import subprocess
import numpy as np

def get_git_hash(repo_dir="."):
    """ obtain the repo's hash version (a short hash string)
    """
    return subprocess.check_output(["git", "describe", "--always"], cwd=repo_dir).strip().decode()


def get_tensor_hash(tensor):
    """ obtain the hash code for torch.Tensor or numpy.ndarray object
    """
    assert isinstance(tensor, (torch.Tensor, np.ndarray))
    array = tensor
    if isinstance(tensor, torch.Tensor):
        array = array.cpu().detach().numpy() # object supporting the buffer API required, so we convert to numpy array
    hash_code = hashlib.sha1(array).hexdigest()
    return hash_code

