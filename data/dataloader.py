import os
from typing import Tuple
import logging

import yaml
from torch.utils.data import DataLoader

from data.create_dataset import split_data
from data.dataset import DRDataset

CONFIG_PATH = "../configs/"
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/data_size.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
log.addHandler(fh)


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


def create_data_loaders(csv_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(csv_name)
    log.debug(len(x_train))
    log.debug(len(x_val))
    log.debug(len(x_test))
    train_data = DRDataset(x_train, y_train, tfms=1)
    val_data = DRDataset(x_val, y_val, tfms=0)
    test_data = DRDataset(x_test, y_test, tfms=0)

    # data loaders
    train_loader = DataLoader(
        train_data, batch_size=config["dataloader"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=config["dataloader"]["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=config["dataloader"]["batch_size"], shuffle=False
    )

    return train_loader, val_loader, test_loader
