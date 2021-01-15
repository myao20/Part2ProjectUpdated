import os
from typing import Tuple
import logging
import numpy as np

import pandas as pd
import yaml
from imutils import paths
from sklearn.model_selection import train_test_split

# TODO: change this to be an absolute path e.g. when running train.py
CONFIG_PATH = "../configs/"


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/create_dataset.log')
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


def load_labels(labels_path: str) -> pd.DataFrame:
    csv_path = os.path.join(labels_path, "trainLabels.csv")
    return pd.read_csv(csv_path)


config = load_config("config.yaml")


def make_csv(image_paths: str, num_each_class: int, csv_name: str) -> None:
    labels = load_labels(config["data_directory"])
    # turn 5 classes into 2
    labels.loc[labels["level"] <= 1, "level"] = 0  # no-DR if level 0, 1
    labels.loc[labels["level"] > 1, "level"] = 1  # DR if level > 1
    image_paths = list(paths.list_images(image_paths))
    num0, num1 = 0, 0
    levels = []
    images = []
    for image_path in image_paths:
        image = os.path.basename(image_path)
        image = os.path.splitext(image)[0]

        level = labels[labels["image"] == image].level.to_numpy()[0]
        if level == 0:
            if num0 >= num_each_class:
                continue
            else:
                num0 += 1
        elif level == 1:
            if num1 >= num_each_class:
                continue
            else:
                num1 += 1
        levels.append(level)
        images.append(image_path)

    df_dict = {'Image_Path': images, 'has_DR': levels}

    df = pd.DataFrame(df_dict)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(csv_name, index=False)
    log.info(csv_name)
    log.info(f'Number in 0 class: {num0}')
    log.info(f'Number in 1 class: {num1}')


def split_data(csv_name: str) -> Tuple:
    df = pd.read_csv(os.path.join(config["data_directory"], csv_name))
    X = df.Image_Path.values
    y = df.has_DR.values
    test_val_size = (
            config["dataset"]["split"]["ratio_test"]
            + config["dataset"]["split"]["ratio_valid"]
    )
    try:
        (x_train, x_test_val, y_train, y_test_val) = train_test_split(
            X, y, test_size=test_val_size, random_state=config["seed"]
        )
        (x_val, x_test, y_val, y_test) = train_test_split(
            x_test_val,
            y_test_val,
            test_size=config["dataset"]["split"]["ratio_test"] / test_val_size,
            random_state=config["seed"],
        )
    except ValueError:
        x_train, y_train = X, y
        x_val, x_test, y_val, y_test = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    make_csv(
        config["image_paths"],
        config["dataset"]["num_each_class"],
        config["dataset"]["csv_name"],
    )


if __name__ == "__main__":
    main()
