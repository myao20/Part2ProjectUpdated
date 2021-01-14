import argparse
import os
import logging
from torch import optim, nn

from data.dataloader import create_data_loaders
from model.base import model
from training.trainer import Trainer

import yaml

log = logging.getLogger("mainLogger")
log.setLevel(logging.INFO)
fh = logging.FileHandler('../logs/bad_images.log')
fh.setLevel(logging.ERROR)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(message)s')
fh.setFormatter(file_formatter)
ch.setFormatter(console_formatter)
log.addHandler(fh)
log.addHandler(ch)

CONFIG_PATH = "../configs/"


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


# launch a run with: python bin/train.py -c $PATH_TO_CONFIG

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
    log.info("Writing results to file")
    trainer.write_logs_to_file()
    log.info("Saving model")
    trainer.save_model_to_file("models/model4.pth")


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

