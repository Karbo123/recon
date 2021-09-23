import resource

def limit_memory(maxsize):
    """ limit cpu memory usage
        example:
            limit_memory("50GB")
        suggestion:
            recommend >= 3GB
    """
    assert any([isinstance(maxsize, int)] + \
               [isinstance(maxsize, str) and maxsize.endswith(end) for end in ("KB", "MB", "GB")]), \
               f"unknown maxsize: {maxsize}"
    if isinstance(maxsize, str):
        for end, p in zip(("KB", "MB", "GB"), (10, 20, 30)):
            if maxsize.endswith(end):
                maxsize = int(maxsize.replace(end, "")) * pow(2, p)

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard)) # Bytes
