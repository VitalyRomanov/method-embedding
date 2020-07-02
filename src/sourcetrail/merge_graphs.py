import pandas as pd
import sys, os
from csv import QUOTE_NONNUMERIC

common_nodes_path = sys.argv[1]
batch_nodes_path = sys.argv[2]

if os.path.isfile(common_nodes_path):
    common_nodes = pd.read_csv(common_nodes_path)
else:
    column_names = ["id", "type", "serialized_name"]
    common_nodes = pd.DataFrame(columns=column_names)

existing_names = set(common_nodes['serialized_name'].values)
# print(existing_names)

records = common_nodes.to_dict(orient="records")

batch_nodes = pd.read_csv(batch_nodes_path)

for ind, row in batch_nodes.iterrows():
    # print(row.serialized_name)
    if row.serialized_name in existing_names: continue

    records.append({
        "id": len(records),
        "type": row.type,
        "serialized_name": row.serialized_name
    })

if len(records) != 0:
    pd.DataFrame(records).to_csv(common_nodes_path, index=False, quoting=QUOTE_NONNUMERIC)
else:
    with open(common_nodes_path, "w") as sink:
        sink.write("id,type,serialized_name\n")
