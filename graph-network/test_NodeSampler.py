import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import RedditDataset
import torch
from NodeSampler import GCNSampling

# load dataset
data = RedditDataset(self_loop=True)
train_nid = torch.LongTensor(np.nonzero(data.train_mask)[0])
features = torch.Tensor(data.features)
in_feats = features.shape[1]
labels = torch.LongTensor(data.labels)
n_classes = data.num_labels

# construct DGLGraph and prepare related data
g = DGLGraph(data.graph, readonly=True)
g.ndata['features'] = features

# number of GCN layers
L = 2
# number of hidden units of a fully connected layer
n_hidden = 64
# dropout probability
dropout = 0.2
# batch size
batch_size = 1000
# number of neighbors to sample
num_neighbors = 4
# number of epochs
num_epochs = 1

# initialize the model and cross entropy loss
model = GCNSampling(in_feats, n_hidden, n_classes, L,
                    torch.nn.functional.relu, dropout)

loss_fcn = torch.nn.CrossEntropyLoss()

# use adam optimizer
optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': 1e-2},
        ], lr=0.01)

for epoch in range(num_epochs):
    i = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, batch_size,
                                                   num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=True,
                                                   num_hops=L,
                                                   seed_nodes=train_nid):
        # When `NodeFlow` is generated from `NeighborSampler`, it only contains
        # the topology structure, on which there is no data attached.
        # Users need to call `copy_from_parent` to copy specific data,
        # such as input node features, from the original graph.
        nf.copy_from_parent()

        pred = model(nf)
        batch_nids = torch.LongTensor(nf.layer_parent_nid(-1))
        batch_labels = labels[batch_nids]
        # cross entropy loss
        loss = loss_fcn(pred, batch_labels)
        loss = loss.sum() / len(batch_nids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # backward
        print("Epoch[{}]: loss {}".format(epoch, loss.item()))
        i += 1
        # You only train the model with 32 mini-batches just for demonstration.
        if i >= 32:
            break