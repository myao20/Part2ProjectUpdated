import os
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from attacks.cw import cw
from attacks.fgsm import fgsm
from typing import List
from torch import nn
import yaml
from torch.utils.data import DataLoader

from attacks.pgd import pgd
from data.dataloader import create_data_loaders
from model.base import model

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/attacks.log')
fh.setLevel(logging.DEBUG)
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


config = load_config("config.yaml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--model-path",
    help="Path to model to be attacked",
    required=True,
)
parser.add_argument(
    "-f",
    "--filename",
    help="Name of plot file",
    required=True,
)


def test_attack(test_model: nn.Module, test_loader: DataLoader, eps: float, criterion) -> float:
    test_model.eval()
    correct = 0
    dataset_length = len(test_loader.dataset)

    for images, labels in test_loader:
        # images = fgsm(test_model, images, labels, eps, criterion).cuda()
        images = cw(test_model, images, labels).cuda()
        labels = labels.cuda()
        outputs = test_model(images)

        _, pre = torch.max(outputs.data, 1)

        correct += (pre == labels).sum().item()

    accuracy = float(correct) / dataset_length
    return accuracy


def run_attack(test_model: nn.Module, test_loader: DataLoader, epsilons: List[float], criterion) -> List[float]:
    accuracies = []

    for eps in epsilons:
        log.info(f'Epsilon: {eps:.4f}')
        acc = test_attack(test_model, test_loader, eps, criterion)
        log.info(f'Accuracy: {acc:.2f}')
        accuracies.append(acc)
    return accuracies


def plot_results(accuracies: List[float], epsilons: List[float], path_name: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(epsilons, accuracies, "x-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(config["output_path"], path_name))


def main():
    epsilons = [0, 0.2/255, 1/255, 2/255, 3/255, 4/255, 5/255]
    args = parser.parse_args()
    log.info(args.model_path)

    train_loader, val_loader, test_loader = create_data_loaders(
        config["dataset"]["csv_name"]
    )

    test_model = model(
        config["model_to_test"]["pretrained"],
        config["model_to_test"]["requires_grad"],
        config["model_to_test"]["add_layers"]
    ).cuda()

    test_model.load_state_dict(torch.load(args.model_path))
    criterion = nn.BCEWithLogitsLoss()
    # accuracies = run_attack(test_model, test_loader, epsilons, criterion)
    # log.info("Plotting results")
    # plot_results(accuracies, epsilons, args.filename)
    accuracy = test_attack(test_model, test_loader, 0, criterion)
    log.info(f'Accuracy after CW L2 attack: {accuracy:.2f}')
    # TODO: add functionality to save a few perturbed images


if __name__ == "__main__":
    main()
