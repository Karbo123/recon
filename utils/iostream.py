import sys, select
from gorilla import Config

def input_with_timeout(words, timeout=10, end=""):
    """ input with timeout
        see: https://stackoverflow.com/questions/1335507/keyboard-input-with-timeout/2904057#2904057
    """
    print(words, end=end)

    i, o, e = select.select([sys.stdin], [], [], timeout)

    if (i):
        return sys.stdin.readline().strip() # string
    else:
        return None # timeout


def print_cfg(cfg, depth=0):
    """ print the gorilla's Config
        NOTE this function is deprecated in a newer gorilla, so we need to implement it ourselves
    """
    if depth == 0:
        assert isinstance(cfg, Config)
    content = ""
    for key, value in dict(cfg).items():
        content += f"{'    ' * depth}{key}: "
        if isinstance(value, dict):
            content += "\n"
            print_cfg(value, depth + 1)
        else:
            content += f"{value}\n"
    return content

