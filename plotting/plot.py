import os
import matplotlib.pyplot as plt
from typing import List


import yaml

CONFIG_PATH = "../configs/"


def load_config(config_name: str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        my_config = yaml.safe_load(file)
    return my_config


config = load_config("config.yaml")


# TODO: edit path name - refactor
def make_plots(train_loss: List[float], train_accuracy: List[float], val_loss: List[float], val_accuracy: List[float],
               path_name: str) -> None:

    # TODO: put below into separate function which returns a list from a file e.g. train_loss
    # with open('C:/path/numbers.txt') as f:
    # lines = f.read().splitlines()
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color="green", label="train accuracy")
    plt.plot(val_accuracy, color="blue", label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))

# TODO: add main function to make plots from files