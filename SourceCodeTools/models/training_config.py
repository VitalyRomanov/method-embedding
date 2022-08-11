from copy import copy

import yaml


config_specification = {
    "DATASET": {
        "data_path": None,
        "train_frac": 0.9,
        "filter_edges": None,
        "min_count_for_objectives": 5,
        # "packages_file": None,  # partition file
        "self_loops": False,
        "use_node_types": False,
        "use_edge_types": False,
        "no_global_edges": False,
        "remove_reverse": False,
        "custom_reverse": None,
        "restricted_id_pool": None,
        "random_seed": None,
        "subgraph_id_column": "mentioned_in",
        "subgraph_partition": None,
        "partition": None
    },
    "TRAINING": {
        "model_output_dir": None,
        "pretrained": None,
        "pretraining_phase": 0,

        "sampling_neighbourhood_size": 10,
        "neg_sampling_factor": 3,
        "use_layer_scheduling": False,
        "schedule_layers_every": 10,

        "elem_emb_size": 100,
        "embedding_table_size": 200000,

        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,

        "objectives": None,
        "save_each_epoch": False,
        "save_checkpoints": True,
        "early_stopping": False,
        "early_stopping_tolerance": 20,

        "force_w2v_ns": False,
        "use_ns_groups": False,
        "nn_index": "brute",

        "metric": "inner_prod",

        "measure_scores": False,
        "dilate_scores": 200,  # downsample

        "gpu": -1,

        "external_dataset": None,

        "restore_state": False,
    },
    "MODEL": {
        "node_emb_size": 100,
        "h_dim": 100,
        "n_layers": 5,
        "use_self_loop": True,

        "use_gcn_checkpoint": False,
        "use_att_checkpoint": False,
        "use_gru_checkpoint": False,

        'num_bases': 10,
        'dropout': 0.0,

        'activation': "relu",
        # torch.nn.functional.hardswish], #[torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
    },
    "TOKENIZER": {
        "tokenizer_path": None,
    }
}


def default_config():
    config = copy(config_specification)

    assert len(
        set(config["MODEL"].keys()) |
        set(config["TRAINING"].keys()) |
        set(config["DATASET"].keys())
    ) == (
            len(set(config["MODEL"].keys())) +
            len(set(config["TRAINING"].keys())) +
            len(set(config["DATASET"].keys()))
    ), "All parameter names in the configuration should be unique"
    return config


def get_config(**kwargs):
    config = default_config()
    return update_config(config, **kwargs)


def update_config(config, **kwargs):
    recognized_options = set()

    for section, args in config.items():
        for key, value in kwargs.items():
            if key in args:
                args[key] = value
                recognized_options.add(key)

    unrecognized = {key: value for key, value in kwargs.items() if key not in recognized_options}

    if len(unrecognized) > 0:
        raise ValueError(f"Some configuration options are not recognized: {unrecognized}")

    return config


def save_config(config, path):
    yaml.dump(config, open(path, "w"))


def load_config(path):
    return yaml.load(open(path, "r").read(), Loader=yaml.Loader)