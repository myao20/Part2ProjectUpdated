import logging

import numpy as np
import torch
import albumentations
from PIL import Image
from torch.utils.data import Dataset

# TODO: if below to be used often, could put into a function in utils folder
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
log = logging.getLogger(__name__)
log.addHandler(console)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# Now, we can log to the root logger, or any other logger. First the root...
logging.info('Jackdaws love my big sphinx of quartz.')

# Now, define a couple of other loggers which might represent areas in your
# application:

logger1 = logging.getLogger('myapp.area1')

logger1.debug('Quick zephyrs blow, vexing daft Jim.')
logger1.info('How quickly daft jumping zebras vex.')


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
        log.info(i, path)
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = self.y[i]
        return torch.tensor(image, dtype=torch.float), torch.tensor(
            label, dtype=torch.long
        )
