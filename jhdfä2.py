
import os
from glob import glob
cur_path = os.getcwd()
ignore_set = set()#["__init__.py", "count_sourcelines.py"])

loc_list = []

files = {}
for py_file in glob("*.py") + glob("./*/*.py"):

    total_path = os.path.join(py_file)
    loc_list.append((len(open(total_path, "r").read().splitlines())))
    files[py_file] = len(open(total_path, "r").read().splitlines())

print(sum(loc_list))
for k, v in files.items():
    print(k, v)
