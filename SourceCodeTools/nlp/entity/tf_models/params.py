import itertools

from sklearn.model_selection import ParameterGrid

cnn_params_grids = {
        "params": [
            # {
            #     "h_sizes": [10, 10, 10],
            #     "dense_size": 10,
            #     "pos_emb_size": 10,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 10,
            #     "suffix_prefix_buckets": 1000,
            # },
            # {
            #     "h_sizes": [20, 20, 20],
            #     "dense_size": 20,
            #     "pos_emb_size": 20,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 20,
            #     "suffix_prefix_buckets": 1000,
            # },
            {
                "h_sizes": [40, 40, 40],
                "dense_size": 30,
                "pos_emb_size": 30,
                "cnn_win_size": 5,
                "suffix_prefix_dims": 50,
                "suffix_prefix_buckets": 2000,
            },
            #     {
            #     "h_sizes": [80, 80, 80],
            #     "dense_size": 40,
            #     "pos_emb_size": 50,
            #     "cnn_win_size": 7,
            #     "suffix_prefix_dims": 70,
            #     "suffix_prefix_buckets": 3000,
            # }
        ],
        "learning_rate": [0.0001],
        "learning_rate_decay": [0.998]  # 0.991
    }

att_params_grids = {
        "params": [
            {
                "h_sizes": [10, 10, 10],
                "dense_size": 10,
                "pos_emb_size": 10,
                "cnn_win_size": 3,
                "suffix_prefix_dims": 10,
                "suffix_prefix_buckets": 1000,
                "target_emb_dim": 5,
                "mention_emb_dim": 5
            },
            {
                "h_sizes": [20, 20],
                "dense_size": 10,
                "pos_emb_size": 20,
                "cnn_win_size": 3,
                "suffix_prefix_dims": 20,
                "suffix_prefix_buckets": 1000,
                "target_emb_dim": 5,
                "mention_emb_dim": 5
            },
            # {
            #     "h_sizes": [20, 20, 20],
            #     "dense_size": 20,
            #     "pos_emb_size": 20,
            #     "cnn_win_size": 3,
            #     "suffix_prefix_dims": 20,
            #     "suffix_prefix_buckets": 1000,
            #     "target_emb_dim": 5,
            #     "mention_emb_dim": 5
            # },
            # {
            #     "h_sizes": [40, 40, 40],
            #     "dense_size": 30,
            #     "pos_emb_size": 30,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 50,
            #     "suffix_prefix_buckets": 2000,
            #     "target_emb_dim": 15,
            #     "mention_emb_dim": 15
            # },
            # {
            #     "h_sizes": [80, 80, 80],
            #     "dense_size": 40,
            #     "pos_emb_size": 50,
            #     "cnn_win_size": 7,
            #     "suffix_prefix_dims": 70,
            #     "suffix_prefix_buckets": 3000,
            #     "target_emb_dim": 25,
            #     "mention_emb_dim": 25,
            # }
        ],
        "learning_rate": [0.0001, 0.001, 0.00001],
        "learning_rate_decay": [0.998] # 0.991
    }


def flatten(params):
    new_params = {}
    for key, val in params.items():
        if type(val) is dict:
            new_params.update(val)
        else:
            new_params[key] = val
    return new_params


cnn_params = list(
    map(flatten, ParameterGrid(cnn_params_grids))
)

att_params = list(
    map(flatten, ParameterGrid(att_params_grids))
)
