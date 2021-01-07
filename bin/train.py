import argparse
import os
import logging
from torch import optim, nn

from data.dataloader import create_data_loaders
from model.base import model
from training.trainer import Trainer

import yaml

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
fh = logging.FileHandler('logfile.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

CONFIG_PATH = "../configs/"


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


# launch a run with: python bin/train.py -c $PATH_TO_CONFIG
# Todo use typehinting
# Todo setup logging


def main():
    # Create Data
    train_loader, val_loader, test_loader = create_data_loaders(
        config["dataset"]["csv_name"]
    )

    # Create Model
    model_fine_tune_added_layers = model(
        pretrained=config["model"]["pretrained"],
        requires_grad=config["model"]["requires_grad"],
        add_layers=config["model"]["add_layers"]
    ).cuda()

    # Create loss
    criterion = nn.BCEWithLogitsLoss()

    # Create optimizer
    optimizer_fine_tune_added_layers = optim.SGD(
        model_fine_tune_added_layers.parameters(),
        lr=config["training"]["optimizer"]["learning_rate"],
        momentum=config["training"]["optimizer"]["momentum"]
    )

    # Create Trainer
    trainer = Trainer(
        model=model_fine_tune_added_layers,
        optimizer=optimizer_fine_tune_added_layers,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
    )

    # train
    log.info("Starting to train the model")
    trainer.train_model()

    # Write logs
    # trainer.write_logs_to_file()
    #
    # trainer.save_model_to_file("models/model2000.pth")


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="../configs/",
        help="Path to project config file",
        required=False,
    )
    args = parser.parse_args()

    parser.parse_args()
    config = load_config("config.yaml")

    main()

