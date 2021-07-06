import sys, select

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

