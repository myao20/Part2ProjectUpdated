import os
import yaml
from torch.utils.data import DataLoader

from data.create_dataset import split_data
from data.dataset import DRDataset

CONFIG_PATH = "../configs/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


def create_data_loaders(csv_name):
    x_train, y_train, x_val, x_test, y_val, y_test = split_data(csv_name)
    train_data = DRDataset(x_train, y_train, tfms=1)
    val_data = DRDataset(x_val, y_val, tfms=0)
    test_data = DRDataset(x_test, y_test, tfms=0)

    # data loaders
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
