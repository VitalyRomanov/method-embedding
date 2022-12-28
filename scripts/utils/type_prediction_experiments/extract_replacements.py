import pandas as pd
import json

data = {"id": set()}

def from_file(path):
    with open(path, "r") as source:
        for line in source:
            entry = json.loads(line)
            for _, _, r in entry["replacements"]:
                data["id"].add(r)

from_file("var_misuse_seq_test.json")
from_file("var_misuse_seq_val.json")
from_file("var_misuse_seq_train.json")

data["id"] = list(data["id"])

pd.DataFrame(data).to_csv("inference_ids.csv", index=False)