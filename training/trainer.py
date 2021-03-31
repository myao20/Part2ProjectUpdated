import logging
import os
import time
import numpy as np
import torch
from torch import nn
from typing import Tuple

from attacks.fgsm import fgsm
from utils.utils import save_model, write_list_to_file

import yaml
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/adv_train.log')
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

        self.num_epochs = config["training"]["num_epochs"]

        # Metrics
        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_accuracy = [], []

    def train_model(self, adv_train=False) -> None:
        start = time.time()
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))

            # Perform training and validation iterations
            train_epoch_loss, train_epoch_accuracy = self.train_iteration(adv_train)
            val_epoch_loss, val_epoch_accuracy = self.validate()

            # Log metrics
            self.train_loss.append(train_epoch_loss)
            self.train_accuracy.append(train_epoch_accuracy)
            self.val_loss.append(val_epoch_loss)
            self.val_accuracy.append(val_epoch_accuracy)
        end = time.time()
        log.info(f"Training time: {(end - start) / 60:.3f} minutes for {self.num_epochs} epochs")

    def train_iteration(self, adv_train: bool) -> Tuple[float, float]:
        print("Training ...")
        self.model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        num_attacked = 0

        dataset_length = len(self.train_loader.dataset)
        for data in self.train_loader:
            data, target = data[0].cuda(), data[1].cuda()
            num_images = len(data)

            if adv_train:
                rand_num = np.random.uniform(size=1)[0]
                if rand_num <= config["training"]["prop_adv_train"]:
                    adv_images, _ = fgsm(self.model, data, target, config["training"]["epsilon"], self.criterion)
                    num_attacked = num_attacked + num_images
                    outputs = self.model(adv_images)
                else:
                    outputs = self.model(data)
            else:
                outputs = self.model(data)

            self.optimizer.zero_grad()
            #outputs = self.model(data)
            y = torch.zeros(list(outputs.size())[0], 2)
            y[range(y.shape[0]), target] = 1
            y = y.cuda()
            loss = self.criterion(outputs, y)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == target).sum().item()
            loss.backward()
            self.optimizer.step()

        proportion_attacked = num_attacked / dataset_length
        log.info(f"Proportion of images attacked: {proportion_attacked}")
        train_loss = train_running_loss / dataset_length
        train_accuracy = 100.0 * train_running_correct / dataset_length
        log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

        return train_loss, train_accuracy

    def validate(self) -> Tuple[float, float]:
        print("Validating ...")
        self.model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        dataset_length = len(self.val_loader.dataset)
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
            log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")

            return val_loss, val_accuracy

    def write_logs_to_file(self) -> None:
        write_list_to_file(self.train_loss, "advtrainlossfgsm.txt")
        write_list_to_file(self.train_accuracy, "advtrainaccfgsm.txt")
        write_list_to_file(self.val_loss, "advvallossfgsm.txt")
        write_list_to_file(self.val_accuracy, "advvalaccfgsm.txt")

    def save_model_to_file(self, filename: str) -> None:
        save_model(self.model, filename)
