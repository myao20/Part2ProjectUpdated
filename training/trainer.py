import logging
import os
import time

import torch
from torch import nn
from typing import Tuple

from utils.utils import save_model, write_list_to_file

import yaml
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

CONFIG_PATH = "../configs/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


# Todo type hinting, logging
# Todo add commentary to training funcs


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            criterion,
    ) -> None:
        # Model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.num_epochs = config["num_epochs"]

        # Metrics
        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_accuracy = [], []

    def train_model(self) -> None:
        start = time.time()
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch, self.num_epochs - 1))

            # Perform training and validation iterations
            train_epoch_loss, train_epoch_accuracy = self.train_iteration()
            val_epoch_loss, val_epoch_accuracy = self.validate()

            # Log metrics
            self.train_loss.append(train_epoch_loss)
            self.train_accuracy.append(train_epoch_accuracy)
            self.val_loss.append(val_epoch_loss)
            self.val_accuracy.append(val_epoch_accuracy)
        end = time.time()
        print(f"{(end - start) / 60:.3f} minutes")

    def train_iteration(self) -> Tuple[float, float]:
        print("Training")
        self.model.train()
        train_running_loss = 0.0
        train_running_correct = 0

        dataset_length = len(self.train_loader.dataset)
        for data in self.train_loader:
            data, target = data[0].cuda(), data[1].cuda()
            self.optimizer.zero_grad()
            outputs = self.model(data)
            y = torch.zeros(list(outputs.size())[0], 2)
            y[range(y.shape[0]), target] = 1
            y = y.cuda()
            loss = self.criterion(outputs, y)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == target).sum().item()
            loss.backward()
            self.optimizer.step()

        train_loss = train_running_loss / dataset_length
        train_accuracy = 100.0 * train_running_correct / dataset_length
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

        return train_loss, train_accuracy

    def validate(self) -> Tuple[float, float]:
        print("Validating")
        self.model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        dataset_length = len(self.val_loader.dataset)
        #  val_loader.dataset.__sizeof__()?
        with torch.no_grad():
            for data in self.val_loader:
                data, target = data[0].cuda(), data[1].cuda()
                outputs = self.model(data)
                y = torch.zeros(list(outputs.size())[0], 2)
                y[range(y.shape[0]), target] = 1
                y = y.cuda()
                loss = self.criterion(outputs, y)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                val_running_correct += (preds == target).sum().item()

            val_loss = val_running_loss / dataset_length
            val_accuracy = 100.0 * val_running_correct / dataset_length
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

            return val_loss, val_accuracy

    def write_logs_to_file(self) -> None:
        write_list_to_file(self.train_loss, "trainloss1.txt")
        write_list_to_file(self.train_accuracy, "trainacc1.txt")
        write_list_to_file(self.val_loss, "valloss1.txt")
        write_list_to_file(self.val_accuracy, "valaccuracy1.txt")

    def save_model_to_file(self, filename: str) -> None:
        save_model(self.model, filename)
