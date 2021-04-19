import argparse
import os
import logging
from torch import optim, nn

from data.dataloader import create_data_loaders
from model.base import model
from training.trainer import Trainer
from utils.utils import exit_if_invalid_path

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


config = load_config("config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--attack",
    default="fgsm",
    help="Attack to be applied - choose from 'fgsm', 'pgd', 'cwl2', 'cwli'",
    required=False,
)
parser.add_argument(
    "--train-acc",
    help="Name of file for writing train accuracy logs to",
    required=True,
)
parser.add_argument(
    "--val-acc",
    help="Name of file for writing val accuracy logs to",
    required=True,
)
parser.add_argument(
    "--train-loss",
    help="Name of file for writing train loss logs to",
    required=True,
)
parser.add_argument(
    "--val-loss",
    help="Name of file for writing val accuracy logs to",
    required=True,
)
parser.add_argument(
    "-p",
    "--model-path",
    help="Path to save model to",
    required=True,
)


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
    args = parser.parse_args()

    exit_if_invalid_path(args.model_path)
    exit_if_invalid_path(args.train_loss)
    exit_if_invalid_path(args.train_acc)
    exit_if_invalid_path(args.val_loss)
    exit_if_invalid_path(args.val_acc)

    log.info(f'Attack being applied is: {args.attack}')

    trainer.train_model(adv_train=config["training"]["adv_train"], attack=args.attack)

    log.info("Saving model")
    trainer.save_model_to_file(args.model_path)
    # Write logs
    log.info("Writing results to file")
    trainer.write_logs_to_file(args.train_loss, args.train_acc, args.val_loss, args.val_acc)


if __name__ == "__main__":
    main()

