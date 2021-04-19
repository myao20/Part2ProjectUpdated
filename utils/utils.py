import os
from typing import Any, List

import torch
from torch import nn

import yaml

CONFIG_PATH = "../configs/"


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


def save_model(my_model: nn.Module, path_name: str) -> None:
    print("Saving model...")
    torch.save(my_model.state_dict(), os.path.join(config["output_path"], path_name))


def write_list_to_file(
    my_list: List[Any], file_name: str, path_name: str = "../outputs/raw_data/"
) -> None:
    file_name = os.path.join(path_name, file_name)
    with open(file_name, "w") as f:
        for item in my_list:
            f.write("%s\n" % item)


def path_valid(file_path: str) -> bool:
    dir_path = os.path.dirname(file_path)
    return os.path.isdir(dir_path)
