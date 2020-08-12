import itertools

from sklearn.model_selection import ParameterGrid

import torch

gat_grids = [
    {
        # 'activation': [nn.functional.leaky_relu],
        # 'num_classes': np.unique(labels).size,
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
    # {
    #     'num_layers': [1],
    #     'in_dim': [100],
    #     'num_hidden': [50],
    #     'heads': [[2, 2]],
    #     'feat_drop': [0.3],
    #     'attn_drop': [0.3],
    #     'negative_slope': [0.2],
    #     'residual': [False]
    # },
    # {
    #     'num_layers': [1],
    #     'in_dim': [100],
    #     'num_hidden': [100],
    #     'heads': [[1, 1]],
    #     'feat_drop': [0.3],
    #     'attn_drop': [0.3],
    #     'negative_slope': [0.2],
    #     'residual': [False]
    # },
    # {
    #     'num_layers': [1],
    #     'in_dim': [50],
    #     'num_hidden': [50],
    #     'heads': [[2, 2, 2]],
    #     'feat_drop': [0.3],
    #     'attn_drop': [0.3],
    #     'negative_slope': [0.2],
    #     'residual': [False]
    # },
    # { # more than one layers leads to oversmoothing. node classification is down, but maybe other tasks become better
    #     'num_layers': [2],
    #     'in_dim': [50],
    #     'num_hidden': [50],
    #     'heads': [[2, 2, 2]],
    #     'feat_drop': [0.3],
    #     'attn_drop': [0.3],
    #     'negative_slope': [0.2],
    #     'residual': [False]
    # },{
    #     'num_layers': [3],
    #     'in_dim': [50],
    #     'num_hidden': [50],
    #     'heads': [[2, 2, 2, 2]],
    #     'feat_drop': [0.3],
    #     'attn_drop': [0.3],
    #     'negative_slope': [0.2],
    #     'residual': [False]
    # }
]

rgcn_grids = [
    # {
    #      # 'embed_size': 100,
    #      # 'bias': True,
    #      # 'activation': nn.functional.leaky_relu,
    #      # 'self_loop': False,
    #      # 'dropout': 0.3,
    #     # 'num_classes',
    #     'h_dim': [50],
    #     'num_bases': [-1],
    #     'num_hidden_layers': [1, 2],
    #     'dropout': [0.3],
    #     'use_self_loop': [False]
    # },
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
        'in_dim': [100],
        'num_hidden': [100],
        # 'num_classes': [100]
        'lr': [0.001]
    }
]


gcnsampling_grids = [
    {
        "in_dim": [100],
        "n_hidden": [100],
        "num_classes": [100],
        "n_layers": [2],
        'dropout': [0.3],
        'activation': [torch.nn.functional.hardtanh] #torch.nn.functional.leaky_relu
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
        'activation': [torch.nn.functional.leaky_relu]  # torch.nn.functional.leaky_relu
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

# print(gat_params)
# print(rgcn_params)
