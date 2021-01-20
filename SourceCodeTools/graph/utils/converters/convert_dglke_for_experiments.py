from SourceCodeTools.graph.model.Embedder import Embedder
import torch
import os, sys
import pandas as pd
import numpy as np
from SourceCodeTools.data.sourcetrail.Dataset import get_train_val_test_indices, SourceGraphDataset, load_data
import pickle
import argparse

def isint(val):
    try:
        int(val)
        return True
    except:
        return False

def load_npy(ent_map_path, npy_path):
    np_embs = np.load(npy_path)
    print(np_embs.shape, np_embs.dtype)

    new_embs = []
    ent_map = {}
    with open(ent_map_path) as ents:
        for line in ents:
            els = line.strip().split()
            if len(els) == 2:
                id_, ent = els
            else:
                continue
            id_ = int(id_)
            try:
                ent = int(ent)
            except:
                continue

            ent_map[ent] = len(new_embs)
            new_embs.append(np_embs[id_])

    new_embs = np.array(new_embs)
    return ent_map, new_embs


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nodes_path', dest='nodes_path', default=None,
                    help='Path to the file with nodes')
parser.add_argument('--edges_path', dest='edges_path', default=None,
                    help='Path to the file with edges')
parser.add_argument('--held_path', dest='held_path', default=None,
                    help='Path to the file with holdout edges')
parser.add_argument('--entities_path', dest='entities_path', default=None,
                    help='')
parser.add_argument('--entities_emb_path', dest='entities_emb_path', default=None,
                    help='')
parser.add_argument('--out_path', dest='out_path', default=None,
                    help='')

args = parser.parse_args()


ent_map, new_embs = load_npy(args.entities_path, args.entities_emb_path)

nodes, edges = load_data(args.nodes_path, args.edges_path)
pd.read_csv(args.held_path).to_csv(os.path.join(args.out_path, "held.csv"), index=False)

nodes['global_graph_id'] = nodes['id'].apply(lambda x: ent_map[x])

# splits = get_train_val_test_indices(nodes.index)
from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types
splits = get_train_val_test_indices(nodes.query(f"type_backup == '{node_types[4096]}'").index)


# nodes, edges, held = SourceGraphDataset.holdout(nodes, edges, 0.001)
# nodes['label'] = nodes['type']

from SourceCodeTools.data.sourcetrail.Dataset import create_train_val_test_masks
# def add_splits(nodes, splits):
#     nodes['train_mask'] = False
#     nodes.loc[nodes.index[splits[0]], 'train_mask'] = True
#     nodes['val_mask'] = False
#     nodes.loc[nodes.index[splits[1]], 'val_mask'] = True
#     nodes['test_mask'] = False
#     nodes.loc[nodes.index[splits[2]], 'test_mask'] = True


emb = Embedder(ent_map, new_embs)

if not os.path.isdir(args.out_path):
    os.mkdir(args.out_path)

torch.save(
    {
        "splits": splits
    },
    os.path.join(args.out_path, "state_dict.pt")
)

create_train_val_test_masks(nodes, *splits)

nodes.to_csv(os.path.join(args.out_path, "nodes.csv"), index=False)
edges.to_csv(os.path.join(args.out_path, "edges.csv"), index=False)
# held.to_csv(os.path.join(args.out_path,  "held.csv"), index=False)

pickle.dump([emb], open(os.path.join(args.out_path, "embeddings.pkl"), "wb"))