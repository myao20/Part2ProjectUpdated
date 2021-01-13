import argparse
import logging
import yaml
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from data.dataloader import create_data_loaders
from model.base import model
from torch.utils.data import DataLoader
from typing import Tuple, List


CONFIG_PATH = "../configs/"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/testing.log')
fh.setLevel(logging.DEBUG)
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


def get_predictions(path_to_model: str, test_loader: DataLoader) -> Tuple[List[int], List[int]]:
    test_model = model(
        config["model_to_test"]["pretrained"],
        config["model_to_test"]["requires_grad"],
        config["model_to_test"]["add_layers"]
    ).cuda()

    test_model.load_state_dict(torch.load(path_to_model))
    test_model.eval()
    test_running_correct = 0
    dataset_length = len(test_loader.dataset)
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data, target = data[0].cuda(), data[1].cuda()
            y_true.extend(target)
            outputs = test_model(data)
            # y = torch.zeros(list(outputs.size())[0], 2)
            # y[range(y.shape[0]), target] = 1

            _, preds = torch.max(outputs.data, 1)
            y_pred.extend(preds)
            test_running_correct += (preds == target).sum().item()

        test_accuracy = 100. * test_running_correct / dataset_length
        print(f'Test Acc: {test_accuracy:.2f}')

        return [i.item() for i in y_true], [i.item() for i in y_pred]


def log_metrics(y_true: List[int], y_pred: List[int]) -> None:
    log.info(f'Accuracy: {accuracy_score(y_true, y_pred):.2f}')
    log.info(f'Precision: {precision_score(y_true, y_pred):.2f}')
    log.info(f'Recall: {recall_score(y_true, y_pred):.2f}')
    log.info(f'F1 score: {f1_score(y_true, y_pred):.2f}')
    print(confusion_matrix(y_true, y_pred))


def main():
    # TODO: find a way to get test_loader from the model used - from the instance of 'Trainer'
    train_loader, val_loader, test_loader = create_data_loaders(
        config["dataset"]["csv_name"]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--model-path",
        help="Path to model to be attacked",
        required=True,
    )
    args = parser.parse_args()
    path_to_model = args.model_path
    log.info(path_to_model)
    y_true, y_pred = get_predictions(path_to_model, test_loader)
    log_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()
