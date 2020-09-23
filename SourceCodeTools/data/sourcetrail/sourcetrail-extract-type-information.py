#%%
import pandas as pd
import sys, os

nodes = pd.read_csv(sys.argv[1], escapechar='\\')
edges = pd.read_csv(sys.argv[2])
type_maps = pd.read_csv(sys.argv[3])
out_annotations = sys.argv[4]
out_no_annotations = sys.argv[5]
annotation_type = int(type_maps.query("desc == 'annotation'")['type'])
returns_type = int(type_maps.query("desc == 'returns'")['type'])

# nodes = pd.read_csv("/home/ltv/data/local_run/method-embedding/src/large_scale/envs/common_nodes.csv")
# edges = pd.read_csv("/home/ltv/data/local_run/method-embedding/src/large_scale/envs/common_edges_with_types_with_ast.csv")
# annotation_type = -3
# returns_type = -2

def ensure_connectedness(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Filtering isolated nodes
    :param nodes: DataFrame
    :param edges: DataFrame
    :return:
    """

    print("Filtering isolated nodes. Starting from {} nodes and {} edges...".format(nodes.shape[0], edges.shape[0]),
          end="")
    unique_nodes = set(edges['src'].values.tolist() +
                       edges['dst'].values.tolist())

    nodes = nodes[
        nodes['id'].apply(lambda nid: nid in unique_nodes)
    ]

    print("ending up with {} nodes and {} edges".format(nodes.shape[0], edges.shape[0]))

    return nodes, edges

#%%

annotations = edges.query(f"type == {annotation_type} or type == {returns_type}")

no_annotations = edges.query(f"type != {annotation_type} and type != {returns_type}")

annotations.to_csv(out_annotations, index=False)
no_annotations.to_csv(out_no_annotations, index=False)

#%%

# edges["annotates"] = False
#
# def find_and_mark(edges, path, csource, index):
#     path.append(csource)
#     edges.loc[index, "annotates"] = True
#
#     next_sources = edges.query(f"target_node_id == {csource}")
#     assert len(next_sources) <= 1, "Node participates in several paths"
#
#     for ind, row in next_sources.iterrows():
#         find_and_mark(edges, path, row.source_node_id, ind)
#
#
#
# for ind, row in annotations.iterrows():
#     path = []
#     find_and_mark(edges, path, row.source_node_id, ind)
