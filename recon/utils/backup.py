""" find dependencies of the python config file
"""

import os
import sys
import gorilla
from shutil import copyfile
from os.path import join, dirname, basename, abspath, relpath


def find_dependencies(pypath):
    lines = [l.strip() for l in gorilla.list_from_file(pypath) if l.strip() != ""]
    pkgs = list()
    for l in lines:
        if "import" in l:
            if l.startswith("from") or l.startswith("import"):
                pkgs.append([s.strip() for s in l.split(" ") if s.strip() != ""][1])
    pkgs = [p for p in pkgs if ("cfg" in p)] # must contain "cfg"
    return pkgs


def find_recursive_dependencies(pypath):
    dir_path = dirname(pypath)
    pkgs = find_dependencies(pypath)
    if len(pkgs) > 0:
        ptr = 0
        while True:
            p = pkgs[ptr]
            new_pkgs = find_dependencies(join(dir_path, p + ".py"))
            pkgs.extend(new_pkgs)
            if ptr == len(pkgs) - 1:
                break
            ptr += 1

    return pkgs


def backup_config(log_dir, cfg):
    config_dir = join(log_dir, "backup", "config")
    os.makedirs(config_dir, exist_ok=True)

    other_cfgs = find_recursive_dependencies(cfg.cfg)
    
    # copy py files
    copyfile(cfg.cfg, join(config_dir, basename(cfg.cfg)))
    dir_path = dirname(cfg.cfg)
    for py in other_cfgs: copyfile(join(dir_path, py + ".py"), join(config_dir, py + ".py"))


def backup_cmdinput(log_dir):
    config_dir = join(log_dir, "backup", "config")
    os.makedirs(config_dir, exist_ok=True)

    basic_path = dirname(dirname(dirname(abspath(__file__))))
    cmd_module = relpath(abspath(sys.argv[0]), basic_path)
    cmd_module = cmd_module.replace(".py", "").replace("/", ".")
    
    cmd = "python -m " + cmd_module + " " + " ".join(sys.argv[1:])
    with open(join(config_dir, "cmdinput.txt"), "w") as f:
        f.write("You may input the following command to terminal:\n\n")
        f.write(cmd + "\n\n")
