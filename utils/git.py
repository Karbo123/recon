import subprocess

def get_git_hash(repo_dir="."):
    """ obtain the repo's hash version (a short hash string)
    """
    return subprocess.check_output(["git", "describe", "--always"], cwd=repo_dir).strip().decode()
