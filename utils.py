# import heapq
# import math
import os
# import random
# import struct
# import sys
from typing import List, Tuple, TypeVar

# try:
#     import importlib.resources as pkg_resources
# except ImportError:
#     import importlib_resources as pkg_resources

T = TypeVar("T")


def var_to_grid(array_var: List[T], size: Tuple[int, int]) -> List[List[T]]:
    """convert ownership/policy to grid format such that grid[y][x] is for move with coords x,y"""
    ix = 0
    grid = [[]] * size[1]
    for y in range(size[1] - 1, -1, -1):
        grid[y] = array_var[ix: ix + size[0]]
        ix += size[0]
    return grid


def find_package_resource(path, silent_errors=False):
    global PATHS
    # if path.startswith("katrain"):
    #     if not PATHS.get("PACKAGE"):
    #         try:
    #             with pkg_resources.path("katrain", "gui.kv") as p:
    #                 PATHS["PACKAGE"] = os.path.split(str(p))[0]
    #         except (ModuleNotFoundError, FileNotFoundError, ValueError) as e:
    #             print(f"Package path not found, installation possibly broken. Error: {e}", file=sys.stderr)
    #             return f"FILENOTFOUND/{path}"
    #     return os.path.join(PATHS["PACKAGE"], path.replace("katrain\\", "katrain/").replace("katrain/", ""))
    return os.path.abspath(os.path.expanduser(path))  # absolute path


def json_truncate_arrays(data, lim=20):
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [json_truncate_arrays(d) for d in data]
        if len(data) > lim:
            data = [f"{len(data)} x {type(data[0]).__name__}"]
        return data
    elif isinstance(data, dict):
        return {k: json_truncate_arrays(v) for k, v in data.items()}
    else:
        return data
