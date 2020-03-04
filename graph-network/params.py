import torch.nn as nn

gat_params = {
    'num_layers': [1],
    'in_dim': [100],
    'num_hidden': [50],
    # 'num_classes': np.unique(labels).size,
    'heads': [[4, 4]],
    'activation': [nn.functional.leaky_relu],
    'feat_drop': [0.3],
    'attn_drop': [0.3],
    'negative_slope': [0.2],
    'residual': [False]
}

rgcn_params = {
     # 'embed_size': 100,
     # 'bias': True,
     # 'activation': nn.functional.leaky_relu,
     # 'self_loop': False,
     # 'dropout': 0.3,
    'h_dim': [50],
    # 'num_classes',
    'num_bases': [-1],
    'num_hidden_layers': [1],
    'dropout': [0.3],
    'use_self_loop': [False]
}