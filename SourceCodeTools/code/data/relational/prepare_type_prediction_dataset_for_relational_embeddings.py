import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from SourceCodeTools.code.ast.python_ast2 import PythonSharedNodes
from SourceCodeTools.code.data.DBStorage import SQLiteStorage
from SourceCodeTools.nlp.entity.utils.data import read_json_data


def prepare(data, dataset_db, out_path):
    new_data = []
    for ind, (text, entry) in enumerate(tqdm(data)):
        repls = set(r[2] for r in entry["replacements"])

        filtered_repls = dataset_db.query(f"""
        select nodes.id as id, type_desc from 
        nodes
        join
        node_types on nodes.type = node_types.type_id
        where nodes.id in ({','.join(map(str, repls))}) and 
        type_desc not in ('{"','".join(PythonSharedNodes.python_token_types)}') and 
        type_desc not in ('module','global_variable','non_indexed_symbol','class','function','class_field','class_method','subword')
        """)

        top_mention = dataset_db.query(f"""
        select mentioned_in, count(edges.id) as count 
        from 
        edges
        join edge_hierarchy on edges.id = edge_hierarchy.id
        join edge_types on edges.type = edge_types.type_id
        where
        (edges.src in ({','.join(map(str, filtered_repls["id"]))}) or
        edges.dst in ({','.join(map(str, filtered_repls["id"]))})) and
        type_desc not in ('defines','uses','imports','calls','uses_type','inheritance','defined_in','used_by','imported_by','called_by','type_used_by','inherited_by') and
        mentioned_in is not null
        group by mentioned_in
        order by count desc 
        """)

        if not (len(top_mention) > 0 and len(top_mention) < 10):
            print(f"No mentions found or too many: {ind}, {len(top_mention)}")
        look_for_mention = top_mention["mentioned_in"]

        edges = dataset_db.query(f"""
        select edges.id as id, type_desc as type, edges.src as src, edges.dst as dst 
        from 
        edges
        join edge_hierarchy on edges.id = edge_hierarchy.id 
        join edge_types on edges.type = edge_types.type_id
        where mentioned_in in ({','.join(map(str, look_for_mention))})
        """)

        nodes_in_slice = set(edges["src"]) | set(edges["dst"])

        nodes = dataset_db.query(f"""
        select nodes.id as id, type_desc as type, nodes.name as name 
        from 
        nodes
        join node_types on nodes.type = node_types.type_id 
        where id in ({','.join(map(str, nodes_in_slice))}) 
        """)

        entry["nodes"] = nodes
        entry["edges"] = edges

        new_data.append((
            text, entry
        ))

    torch.save(new_data, out_path)




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
    prepare(train_data, dataset_db, output.joinpath("train_relational_type_pred.pt"))
    prepare(test_data, dataset_db, output.joinpath("test_relational_type_pred.pt"))



if __name__ == "__main__":
    main()