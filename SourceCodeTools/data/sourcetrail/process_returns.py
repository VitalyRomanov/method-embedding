import pandas as pd
import os, sys

nodes_path = sys.argv[1]
edges_path = sys.argv[2]
annotation_edges_path = sys.argv[3]
type_maps = pd.read_csv(sys.argv[4])
fname_type = int(type_maps.query("desc == 'fname'")['type'])
return_type = int(type_maps.query("desc == 'returns'")['type'])

nodes = pd.read_csv(nodes_path).astype({"serialized_name": "str"}).rename({"id": "source_node_id"}, axis=1)
edges = pd.read_csv(edges_path)
annotations = pd.read_csv(annotation_edges_path)

only_fnames = edges.query(f"type == {fname_type}")
only_returns = annotations.query(f"type == {return_type}")

only_returns = only_returns.merge(nodes, on="source_node_id")
only_returns['type_name'] = only_returns['serialized_name'].apply(lambda x: x.split("[")[0].split(".")[-1].strip("\""))

new_edges = []

for ind, row in only_returns.iterrows():
    fid = row['target_node_id']
    fname_edges = only_fnames.query(f"source_node_id == {fid}")

    assert len(fname_edges) == 1
    fnode = fname_edges.iloc[0]['target_node_id']

    new_edges.append({"src": fnode, "dst": row['type_name']})

pd.DataFrame(new_edges).to_csv(os.path.join(os.path.dirname(annotation_edges_path), "return_types_decoded.csv"), index=False)