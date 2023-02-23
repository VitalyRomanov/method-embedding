from copy import copy

import yaml


def default_config(config_specification):
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


def update_config(config, **kwargs):
    recognized_options = set()

    for section, args in config.items():
        for key, value in kwargs.items():
            if key in args:
                recognized_options.add(key)
                if value is not None:
                    args[key] = value

    unrecognized = {key: value for key, value in kwargs.items() if key not in recognized_options}

    if len(unrecognized) > 0:
        raise ValueError(f"Some configuration options are not recognized: {unrecognized}")

    return config


def save_config(config, path):
    yaml.dump(config, open(path, "w"))


def load_config(path):
    return yaml.load(open(path, "r").read(), Loader=yaml.Loader)