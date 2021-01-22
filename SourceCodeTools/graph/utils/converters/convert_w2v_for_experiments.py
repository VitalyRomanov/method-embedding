from SourceCodeTools.graph.model.Embedder import Embedder
import torch
import os, sys
import numpy as np
from SourceCodeTools.code.data.sourcetrail.Dataset import get_train_val_test_indices, load_data
import pickle


def isint(val):
    try:
        int(val)
        return True
    except:
        return False

def load_w2v(path):
    id_map = {}
    vecs = []
    with open(emb_path) as vectors:
        n_vectors, n_dims = map(int, vectors.readline().strip().split())

        for ind in range(n_vectors):
            elements = vectors.readline().strip().split()

            if isint(elements[0]):
                id_ = int(elements[0])
            else:
                continue

            vec = list(map(float, elements[1:]))
            assert len(vec) == n_dims
            id_map[id_] = len(vecs)
            vecs.append(vec)
    vecs = np.array(vecs)
    return id_map, vecs

emb_path = sys.argv[1]
nodes_path = sys.argv[2]
edges_path = sys.argv[3]
out_path = sys.argv[4]

nodes, edges = load_data(nodes_path, edges_path)

# splits = get_train_val_test_indices(nodes.index)
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types
splits = get_train_val_test_indices(nodes.query(f"type_backup == '{node_types[4096]}'").index)

id_map, vecs = load_w2v(emb_path)

nodes['global_graph_id'] = nodes['id'].apply(lambda x: id_map[x])

# nodes, edges, held = SourceGraphDataset.holdout(nodes, edges, 0.001)
# nodes['label'] = nodes['type']

# emb = Embedder.load_word2vec(emb_path)
emb = Embedder(id_map, vecs)

if not os.path.isdir(out_path):
    os.mkdir(out_path)

torch.save(
    {
        "splits": splits
    },
    os.path.join(out_path, "state_dict.pt")
)

from SourceCodeTools.code.data.sourcetrail.Dataset import create_train_val_test_masks
create_train_val_test_masks(nodes, *splits)

nodes.to_csv(os.path.join(out_path, "nodes.csv"), index=False)
edges.to_csv(os.path.join(out_path, "edges.csv"), index=False)
# held.to_csv(os.path.join(out_path,  "held.csv"), index=False)

pickle.dump([emb], open(os.path.join(out_path, "embeddings.pkl"), "wb"))