import logging

import numpy as np
import torch
import albumentations
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger("mainLogger")


class DRDataset(Dataset):
    def __init__(self, path, labels, tfms=None):
        self.X = path
        self.y = labels
        # apply augmentations
        if tfms == 0:  # if validating
            self.aug = albumentations.Compose(
                [
                    albumentations.Resize(224, 224, always_apply=True),
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        always_apply=True,
                    ),
                ]
            )
        else:  # if training
            self.aug = albumentations.Compose(
                [
                    albumentations.Resize(224, 224, always_apply=True),
                    albumentations.HorizontalFlip(p=1.0),
                    albumentations.ShiftScaleRotate(
                        shift_limit=0.3, scale_limit=0.3, rotate_limit=30, p=1.0
                    ),
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        always_apply=True,
                    ),
                ]
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        image = Image.open(self.X[i])
        path = self.X[i]
        log.info(path)
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(
            label, dtype=torch.long
        )
