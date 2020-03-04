from sklearn.model_selection import ParameterGrid
import itertools

gat_grids = [{
    # 'activation': [nn.functional.leaky_relu],
    # 'num_classes': np.unique(labels).size,
    'num_layers': [1],
    'in_dim': [50],
    'num_hidden': [50],
    'heads': [[2, 2]],
    'feat_drop': [0.3],
    'attn_drop': [0.3],
    'negative_slope': [0.2],
    'residual': [False]
},{
    'num_layers': [2],
    'in_dim': [50],
    'num_hidden': [50],
    'heads': [[2, 2, 2]],
    'feat_drop': [0.3],
    'attn_drop': [0.3],
    'negative_slope': [0.2],
    'residual': [False]
},{
    'num_layers': [3],
    'in_dim': [50],
    'num_hidden': [50],
    'heads': [[2, 2, 2, 2]],
    'feat_drop': [0.3],
    'attn_drop': [0.3],
    'negative_slope': [0.2],
    'residual': [False]
}
]

rgcn_grids = [{
     # 'embed_size': 100,
     # 'bias': True,
     # 'activation': nn.functional.leaky_relu,
     # 'self_loop': False,
     # 'dropout': 0.3,
    # 'num_classes',
    'h_dim': [50],
    'num_bases': [-1],
    'num_hidden_layers': [1, 2, 3],
    'dropout': [0.3],
    'use_self_loop': [False]
}]

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

# print(gat_params)
# print(rgcn_params)