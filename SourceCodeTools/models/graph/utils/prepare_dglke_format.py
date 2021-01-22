import os
import pandas as pd
from SourceCodeTools.code.data.sourcetrail.Dataset import SourceGraphDataset, load_data, compact_property

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nodes_path', dest='nodes_path', default=None,
                    help='Path to the file with nodes')
parser.add_argument('--edges_path', dest='edges_path', default=None,
                    help='Path to the file with edges')
parser.add_argument('--fname_path', dest='fname_path', default=None,
                    help='')
parser.add_argument('--varuse_path', dest='varuse_path', default=None,
                    help='')
parser.add_argument('--apicall_path', dest='apicall_path', default=None,
                    help='')
parser.add_argument('--out_path', dest='out_path', default=None,
                    help='')

args = parser.parse_args()

nodes, edges = load_data(args.nodes_path, args.edges_path)
node2graph_id = compact_property(nodes['id'])
nodes['global_graph_id'] = nodes['id'].apply(lambda x: node2graph_id[x])

nodes, edges, held = SourceGraphDataset.holdout(nodes, edges, 0.005)
edges.to_csv(os.path.join(args.out_path, "edges_train.csv"), index=False)
held.to_csv(os.path.join(args.out_path, "held.csv"), index=False)

edges = edges.astype({"src": 'str', "dst": "str", "type": 'str'})[['src', 'dst', 'type']]

node_ids = set(nodes['id'].unique())

if args.fname_path is not None:
    fname = pd.read_csv(args.fname_path).astype({"src": 'int32', "dst": "str"})
    fname['type'] = 'fname'

    fname = fname[
        fname['src'].apply(lambda x: x in node_ids)
    ]

    edges = pd.concat([edges, fname])

if args.varuse_path is not None:
    varuse = pd.read_csv(args.varuse_path).astype({"src": 'int32', "dst": "str"})
    varuse['type'] = 'varuse'

    varuse = varuse[
        varuse['src'].apply(lambda x: x in node_ids)
    ]

    edges = pd.concat([edges, varuse])

if args.apicall_path is not None:
    apicall = pd.read_csv(args.apicall_path).astype({"src": 'int32', "dst": "int32"})
    apicall['type'] = 'nextcall'

    apicall = apicall[
        apicall['src'].apply(lambda x: x in node_ids)
    ]

    edges = pd.concat([edges, apicall])

# splits = get_train_test_val_indices(edges.index, train_frac=0.6)

nodes['label'] = nodes['type']

if not os.path.isdir(args.out_path):
    os.mkdir(args.out_path)

nodes.to_csv(os.path.join(args.out_path, "nodes.csv"), index=False)
edges.to_csv(os.path.join(args.out_path, "edges_train_dglke.tsv"), index=False, header=False, sep="\t")
edges[['src','dst']].to_csv(os.path.join(args.out_path, "edges_train_node2vec.csv"), index=False, header=False, sep=" ")
# edges.iloc[splits[0]].to_csv(os.path.join(args.out_path, "edges_train.csv"), index=False, header=False, sep="\t")
# edges.iloc[splits[1]].to_csv(os.path.join(args.out_path, "edges_val.csv"), index=False, header=False, sep="\t")
# edges.iloc[splits[2]].to_csv(os.path.join(args.out_path, "edges_test.csv"), index=False, header=False, sep="\t")
held[['src','dst','type']].to_csv(os.path.join(args.out_path, "held_dglkg.tsv"), index=False, sep='\t', header=False)
held[['src','dst']].to_csv(os.path.join(args.out_path, "held_node2vec.csv"), index=False, sep=' ', header=False)