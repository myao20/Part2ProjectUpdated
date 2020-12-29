import time
import os

from torch import optim, nn

from data.dataloader import create_data_loaders
from model.base import model
from training.trainer import validate, fit

import yaml

CONFIG_PATH = "../configs/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


def train_model(train_loader, val_loader, my_model, optimizer, num_epochs, criterion):
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        train_epoch_loss, train_epoch_accuracy = fit(my_model, train_loader, optimizer, criterion)
        val_epoch_loss, val_epoch_accuracy = validate(my_model, val_loader, criterion)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
    end = time.time()
    print(f"{(end - start) / 60:.3f} minutes")
    return train_loss, train_accuracy, val_loss, val_accuracy


def main():
    criterion = nn.BCEWithLogitsLoss()
    model_fine_tune_added_layers = model(pretrained=True, requires_grad=True, add_layers=True).cuda()
    optimizer_fine_tune_added_layers = optim.SGD(model_fine_tune_added_layers.parameters(),
                                                 lr=config["training"]["optimizer"]["learning_rate"],
                                                 momentum=config["training"]["optimizer"]["momentum"])
    train_loader, val_loader, test_loader = create_data_loaders(config["dataset"]["csv_name"])
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(train_loader, val_loader,
                                                                     model_fine_tune_added_layers,
                                                                     optimizer=optimizer_fine_tune_added_layers,
                                                                     num_epochs=config["training"]["num_epochs"],
                                                                     criterion=criterion)
    # TODO: output graphs? - find some sort of tracker
    return train_loss, train_accuracy, val_loss, val_accuracy


if __name__ == "__main__":
    main()

