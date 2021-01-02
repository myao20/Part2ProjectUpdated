import os
from typing import Any, List

import torch


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
