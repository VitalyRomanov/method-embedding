import itertools

from sklearn.model_selection import ParameterGrid

import torch

gat_grids = [
    {
        'num_layers': [1],
        'in_dim': [50],
        'num_hidden': [50],
        'heads': [[2, 2]],
        'feat_drop': [0.3],
        'attn_drop': [0.3],
        'negative_slope': [0.2],
        'residual': [False],
        'activation': [torch.nn.functional.leaky_relu], #torch.nn.functional.leaky_relu
        'lr': [0.01]
    },
]

rgcn_grids = [
    {
        'h_dim': [100],
        'num_bases': [-1],
        'num_hidden_layers': [2],
        'dropout': [0.3],
        'use_self_loop': [False],
        'activation': [torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
        'lr': [0.001]
    }
]

ggnn_grids = [
    {
        'n_steps': [8],
        'in_dim': [200],
        'num_hidden': [200],
        # 'num_classes': [100]
        'lr': [0.005],
        'activation': [torch.nn.functional.hardtanh]
    }
]


gcnsampling_grids = [
    {
        "in_dim": [100],
        "n_hidden": [100],
        # "num_classes": [100],
        "n_layers": [2],
        'dropout': [0.3],
        'activation': [torch.nn.functional.hardtanh] #torch.nn.functional.leaky_relu
    }
]

rgcnsampling_grids = [
    {
        'h_dim': [100],
        'num_bases': [-1],
        'num_hidden_layers': [1],
        'dropout': [0.3],
        'use_self_loop': [False],
        'activation': [torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
        'lr': [1e-4]
    }
]

rggan_grids = [
    {
        'h_dim': [100],
        'num_bases': [10],
        'num_steps': [9],
        'dropout': [0.2],
        'use_self_loop': [False],
        'activation': [torch.tanh], # torch.nn.functional.hardswish], #[torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
        'lr': [1e-3], # 1e-4]
    }
]

gatsampling_grids = [
    {
        'num_layers': [1],
        'in_dim': [100],
        'num_hidden': [50],
        'heads': [[2, 2]],
        'feat_drop': [0.3],
        'attn_drop': [0.3],
        'negative_slope': [0.2],
        'residual': [False],
        'activation': [torch.nn.functional.leaky_relu],  # torch.nn.functional.leaky_relu
        'lr': [0.01]
    },
]

gat_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in gat_grids]
    )
)

rgcn_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rgcn_grids]
    )
)

rgcnsampling_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rgcnsampling_grids]
    )
)

ggnn_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in ggnn_grids]
    )
)

gcnsampling_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in gcnsampling_grids]
    )
)

gatsampling_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in gatsampling_grids]
    )
)


rggan_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rggan_grids]
    )
)
