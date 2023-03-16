
import yaml
from collections import OrderedDict


def parse_unknown_args(args_unknown):
    """ try to parse unknown args
    """

    try_parse = lambda s: yaml.safe_load(f"item: {s}").get("item") # using yaml

    args = OrderedDict()
    for item in args_unknown:
        if item.startswith("--"):
            args[item[2:]] = None
        elif len(args) > 0:
            key = next(reversed(args.keys()))
            if args[key] is None:
                args[key] = try_parse(item)
            elif isinstance(args[key], list):
                args[key].append(try_parse(item))
            else:
                args[key] = [args[key], try_parse(item)]
    
    return args
