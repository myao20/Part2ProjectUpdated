import argparse
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging

import yaml

CONFIG_PATH = "../configs/"

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
fh = logging.FileHandler('../logs/plotting.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-acc",
    help="Path to train accuracy file",
    required=True,
)
parser.add_argument(
    "--val-acc",
    help="Path to validation accuracy file",
    required=True,
)
parser.add_argument(
    "--train-loss",
    help="Path to train loss file",
    required=True,
)
parser.add_argument(
    "--val-loss",
    help="Path to validation loss file",
    required=True,
)
parser.add_argument(
    "--acc-filename",
    help="Name of accuracy plot",
    required=True,
)
parser.add_argument(
    "--loss-filename",
    help="Name of loss plot",
    required=True,
)


def lists_from_files(path_to_file1: str, path_to_file2: str) -> Tuple[List[float], List[float]]:
    with open(path_to_file1) as f:
        l1 = f.read().splitlines()
        l1 = [float(i) for i in l1]
    with open(path_to_file2) as f:
        l2 = f.read().splitlines()
        l2 = [float(i) for i in l2]
    return l1, l2


def make_accuracy_plots(train_accuracy: List[float], val_accuracy: List[float], path_name: str) -> None:
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color="green", label="train accuracy")
    plt.plot(val_accuracy, color="blue", label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))


def make_loss_plots(train_loss: List[float], val_loss: List[float], path_name: str) -> None:
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))


def plot_accuracy(train_accuracy_path: str, val_accuracy_path: str, acc_filename: str) -> None:
    train_accuracy, val_accuracy = lists_from_files(train_accuracy_path, val_accuracy_path)
    # make_accuracy_plots(train_accuracy, val_accuracy, "plots/advfgsm_acc.png")
    make_accuracy_plots(train_accuracy, val_accuracy, acc_filename)


def plot_loss(train_loss_path: str, val_loss_path: str, loss_filename: str) -> None:
    train_loss, val_loss = lists_from_files(train_loss_path, val_loss_path)
    # make_loss_plots(train_loss, val_loss, "plots/advfgsm_loss.png")
    make_loss_plots(train_loss, val_loss, loss_filename)


def main():
    args = parser.parse_args()
    log.info("Making accuracy plots")
    plot_accuracy(args.train_acc, args.val_acc, args.acc_filename)
    log.info("Making loss plots")
    plot_loss(args.train_loss, args.val_loss, args.loss_filename)


if __name__ == "__main__":
    main()
