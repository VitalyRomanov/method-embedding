import itertools

from sklearn.model_selection import ParameterGrid


rgcn_grids = [
    {
        'h_dim': [100],
        'node_emb_size': [100],
        'elem_emb_size': [100],
        'num_bases': [10],
        'n_layers': [5],
        'dropout': [0.3],
        'use_self_loop': [True],
        'activation': ["leaky_relu"],  # leaky_relu hardtanh
        'lr': [1e-4]
    }
]


rggan_grids = [
    {
        'h_dim': [100],
        'node_emb_size': [100],
        'elem_emb_size': [100],
        'num_bases': [10],
        'n_layers': [5],
        'dropout': [0.2],
        'use_self_loop': [True],
        'activation': ["leaky_relu"], # "hardswish" "hardtanh" "tanh"
        'lr': [1e-3],
    }
]


rgcn_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rgcn_grids]
    )
)


rggan_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rggan_grids]
    )
)
