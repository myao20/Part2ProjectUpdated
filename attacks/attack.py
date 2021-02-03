import os
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from attacks.cw import cw
from attacks.fgsm import fgsm
from typing import List, Any, Tuple
from torch import nn
import yaml
from torch.utils.data import DataLoader

from attacks.pgd import pgd
from data.dataloader import create_data_loaders
from model.base import model

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/attacks.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
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
    required=False,
)
parser.add_argument(
    "-af",
    "--adv_filename",
    help="Name of adversarial images file",
    required=False,
)
parser.add_argument(
    "-of",
    "--orig_filename",
    help="Name of original images file",
    required=False,
)
parser.add_argument(
    "-pf",
    "--perturb_filename",
    help="Name of perturbations file",
    required=False,
)


def get_adv_indices(num: int, init_pre, new_pre, labels) -> np.ndarray:
    initial_matching_preds = (init_pre == labels).detach().cpu().numpy()
    matching_indices = np.where(initial_matching_preds == True)
    changed_preds = (init_pre == new_pre).detach().cpu().numpy()
    changed_indices = np.where(changed_preds == False)
    adv_indices = np.intersect1d(matching_indices, changed_indices)
    return adv_indices[:max(5, num)]


def test_attack(test_model: nn.Module, test_loader: DataLoader, eps: float, criterion) -> Tuple[
                float, List[Tuple[Any, Any, Any]], List[Tuple[Any, Any, Any]], List[Tuple[Any, Any, Any]]]:
    test_model.eval()
    correct = 0
    dataset_length = len(test_loader.dataset)
    adv_examples = []
    orig_examples = []
    perturbations = []

    for images, labels in test_loader:
        adv_images, outputs = pgd(test_model, images, labels, eps, criterion)
        # adv_images, outputs = cw(test_model, images, labels)
        _, init_preds = torch.max(outputs.data, 1)
        labels = labels.cuda()
        outputs = test_model(adv_images)

        _, new_preds = torch.max(outputs.data, 1)
        if len(adv_examples) < 5:
            adv_example_indices = get_adv_indices(5 - len(adv_examples), init_preds, new_preds, labels)
            for i in adv_example_indices:
                log.info(f'Adversarial example index: {i}')
                adv_ex = adv_images[i].squeeze().detach().cpu().numpy()
                adv_examples.append((init_preds[i].item(), new_preds[i].item(), adv_ex))
                orig_ex = images[i].squeeze().detach().cpu().numpy()
                orig_examples.append((init_preds[i].item(), new_preds[i].item(), orig_ex))
                perturbation = adv_ex - orig_ex
                perturbations.append((init_preds[i].item(), new_preds[i].item(), perturbation))

        correct += (new_preds == labels).sum().item()

    accuracy = float(correct) / dataset_length
    return accuracy, adv_examples, orig_examples, perturbations


def run_attack(test_model: nn.Module, test_loader: DataLoader, epsilons: List[float], criterion) -> Tuple[
                List[float], List[List[Tuple[Any, Any, Any]]], List[List[Tuple[Any, Any, Any]]],
                List[List[Tuple[Any, Any, Any]]]]:
    accuracies = []
    examples = []
    orig_examples = []
    perturbations = []

    for eps in epsilons:
        log.info(f'Epsilon: {eps:.4f}')
        acc, ex, orig, perturbation = test_attack(test_model, test_loader, eps, criterion)
        log.info(f'Accuracy: {acc:.2f}')
        accuracies.append(acc)
        examples.append(ex)
        orig_examples.append(orig)
        perturbations.append(perturbation)
    return accuracies, examples, orig_examples, perturbations


def plot_results(accuracies: List[float], epsilons: List[float], path_name: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(epsilons, accuracies, "x-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(config["output_path"], path_name))


def save_example_images(epsilons: List[float], examples: List[List[Tuple[Any, Any, Any]]], file_name: str,
                        scale_perturbations: bool):
    # Plot 5 examples of adversarial images for epsilon
    cnt = 0
    plt.figure(figsize=(20, 30))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            if len(examples) > 1:
                plt.subplot(len(epsilons), len(examples[1]), cnt)
            else:
                plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                # TODO: edit below so max 4 decimal places or similar
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig_pred, new_pred, ex = examples[i][j]
            # option to magnify perturbations so they're perceptible
            if scale_perturbations:
                ex = ex*100
            plt.title("{} -> {}".format(orig_pred, new_pred))
            plt.imshow(ex.transpose(2, 1, 0), cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_path"], file_name))


def main():
    epsilons = [0, 0.2 / 255, 1 / 255, 2 / 255, 3 / 255, 4 / 255, 5 / 255]
   # epsilons = [0, 0.002, 0.02]
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
    accuracies, examples, orig_examples, perturbations = run_attack(test_model, test_loader, epsilons, criterion)
    # log.info("Plotting results")
    # plot_results(accuracies, epsilons, args.filename)
    log.info("Saving some adversarial images")
    save_example_images(epsilons, examples, args.adv_filename, False)
    log.info("Saving the original images")
    save_example_images(epsilons, orig_examples, args.orig_filename, False)
    log.info("Saving the perturbations")
    save_example_images(epsilons, perturbations, args.perturb_filename, True)

    # accuracy = test_attack(test_model, test_loader, 0, criterion)
    # log.info(f'Accuracy after CW L2 attack: {accuracy:.2f}')


if __name__ == "__main__":
    main()
