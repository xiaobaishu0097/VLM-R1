import os
from copy import deepcopy

import yaml


def merge(d1, d2):
    BASIC_TYPES = (int, float, str, bool, complex)
    if isinstance(d1, dict) and isinstance(d2, dict):
        for key, value in d2.items():
            if key in d1:
                if not isinstance(d1[key], BASIC_TYPES):
                    d1[key] = merge(d1[key], value)
                #     d1[key] = value
                # else:
                #     d1[key] = merge(d1[key], value)
            else:
                d1[key] = deepcopy(value)
        return d1
    if isinstance(d1, list) and isinstance(d2, list):
        return d1 + d2
    return deepcopy(d2)


def load_config(file_path) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        if "inherit" in config:
            if isinstance(config["inherit"], list):
                for inherit in config["inherit"]:
                    inherit_path = os.path.join(os.path.dirname(file_path), inherit)
                    if os.path.exists(inherit_path):
                        parent = load_config(inherit_path)
                        config = merge(config, parent)
            elif isinstance(config["inherit"], str):
                inherit_path = os.path.join(
                    os.path.dirname(file_path), config["inherit"]
                )
                if os.path.exists(inherit_path):
                    parent = load_config(inherit_path)
                    config = merge(config, parent)
            else:
                raise ValueError("Invalid inherit type")

        return config
