import argparse
import logging
import os
import time

import torch
import yaml
from torch import nn, optim

from data.dataloader import create_data_loaders
from model.base import model
from training.trainer import Trainer
from utils.utils import save_model, write_list_to_file

log = logging.getLogger(__name__)

CONFIG_PATH = "../configs/"


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


# Todo check config parsing from command line. You hould be able to launch a run with: python bin/train.py -c $PATH_TO_CONFIG
# Todo use typehinting
# Todo setup logging


def main():
    # Create Data
    train_loader, val_loader, test_loader = create_data_loaders(
        config["dataset"]["csv_name"]
    )

    # Create Model
    # Todo model params should be configurable from config
    model_fine_tune_added_layers = model(
        pretrained=True, requires_grad=True, add_layers=True
    ).cuda()

    # Create loss
    # Todo this should be configurable from config
    criterion = nn.BCEWithLogitsLoss()

    # Create optimizer
    # Todo optimizer type should be configurable from config (ADAM, etc)
    optimizer_fine_tune_added_layers = optim.SGD(
        model_fine_tune_added_layers.parameters(),
        lr=config["training"]["optimizer"]["learning_rate"],
        momentum=config["training"]["optimizer"]["momentum"],
    )

    # Create Trainer
    trainer = Trainer(
        model=model_fine_tune_added_layers,
        optimizer=optimizer_fine_tune_added_layers,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        config=config["training"],
    )

    # Write logs
    # TODO: output graphs? - find some sort of tracker
    trainer.write_logs_to_file()

    trainer.save_model("models/model2000.pth")
    save_model(model_fine_tune_added_layers, "models/model2000.pth")

    return train_loss, train_accuracy, val_loss, val_accuracy


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="/Users/arianjamasb/github/Part2ProjectUpdated/config/default_config.json",
        help="Path to project config file",
        required=False,
    )
    args = parser.parse_args()

    parser.parse_args()
    config = load_config("config.yaml")

    main()
