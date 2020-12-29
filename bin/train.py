import time
import os
import matplotlib.pyplot as plt

import torch
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


def save_model(my_model, path_name):
    print('Saving model...')
    torch.save(my_model.state_dict(), os.path.join(config["output_path"], path_name))


# TODO: edit path name - refactor, put in different folder - utils?
def make_plots(train_loss, train_accuracy, val_loss, val_accuracy, path_name):

    #with open('C:/path/numbers.txt') as f:
    #lines = f.read().splitlines()
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))


def write_list_to_file(my_list, file_name, path_name="../outputs/raw_data/"):
    file_name = os.path.join(path_name, file_name)
    with open(file_name, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


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
    write_list_to_file(train_loss, "trainloss1.txt")
    write_list_to_file(train_accuracy, "trainacc1.txt")
    write_list_to_file(val_loss, "valloss1.txt")
    write_list_to_file(val_accuracy, "valaccuracy1.txt")

    return train_loss, train_accuracy, val_loss, val_accuracy


if __name__ == "__main__":
    main()

