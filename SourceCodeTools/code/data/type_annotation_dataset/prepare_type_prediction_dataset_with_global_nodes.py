import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from SourceCodeTools.code.ast.python_ast2 import PythonSharedNodes
from SourceCodeTools.code.data.DBStorage import SQLiteStorage
from SourceCodeTools.nlp.entity.utils.data import read_json_data


def prepare(data, dataset_db, out_path):

    with open(out_path, "w") as sink:
        for ind, (text, entry) in enumerate(tqdm(data)):
            repls = set(r[2] for r in entry["replacements"])

            global_nodes = dataset_db.query(f"""
            select nodes.id as id, type_desc, name from 
            nodes
            join
            node_types on nodes.type = node_types.type_id
            where nodes.id in ({','.join(map(str, repls))}) and 
            type_desc in ('module','global_variable','non_indexed_symbol','class','function','class_field','class_method')
            """)

            repl_name = dict(zip(global_nodes["id"], global_nodes["name"]))

            filtered_repls = [repl for repl in entry["replacements"] if repl[2] in global_nodes["id"].tolist()]

            entry["replacements"] = filtered_repls
            entry["replacement_names"] = repl_name

            sink.write(f"{json.dumps((text, entry))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_db")
    parser.add_argument("type_pred_path")
    parser.add_argument("output")

    args = parser.parse_args()

    train_data, test_data = read_json_data(
        args.type_pred_path, normalize=True, allowed=None, include_replacements=True, include_only="entities",
        min_entity_count=0
    )

    dataset_db = SQLiteStorage(args.dataset_db)

    output = Path(args.output)
    prepare(train_data, dataset_db, output.joinpath("train_type_pred_with_global_nodes.json"))
    prepare(test_data, dataset_db, output.joinpath("test_type_pred_with_global_nodes.json"))



if __name__ == "__main__":
    main()