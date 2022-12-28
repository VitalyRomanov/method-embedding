import json
import sys
from collections import defaultdict

import pandas as pd

path = sys.argv[1]

all_types = f"{path}/perform_experiment_all_types.log"
pop_types = f"{path}/perform_experiment_pop_types.log"



def parse_log(path, partition):
    table = defaultdict(list)

    for line in open(path, "r"):
        if line.startswith("{") and not line.startswith("{'ex"):
            line = line.replace("\'", "\"").strip()
            try:
                entry = json.loads(line)
            except:
                print(line)
                sys.exit()
            for k in entry:
                table[f"{partition}_{k}"].append(entry[k])

    return table

all_tbl = parse_log(all_types, "all")
pop_tbl = parse_log(pop_types, "pop")

def print_table(table):
    for key in table:
        print(key, end="\t")
    print()

    for i in range(5):
        for key in table:
            print(table[key][i] * 100, end="\t")
        print()

print_table(all_tbl)
print_table(pop_tbl)