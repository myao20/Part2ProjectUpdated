import logging

import torch.nn as nn
from torchvision import models as models

log = logging.getLogger(__name__)

# Todo base should contain an abstract class for a model. You should keep a separate script for building your desired model from a config.
#  In feats, dropout, num additional layers, activations et should be configurable from config
# Todo use logging, typehinting


def model(pretrained, requires_grad, add_layers):
    base_model = models.resnet50(progress=True, pretrained=pretrained)
    # freeze hidden layers
    if not requires_grad:
        for param in base_model.parameters():
            param.requires_grad = False
    # train the hidden layers
    elif requires_grad:
        for param in base_model.parameters():
            param.requires_grad = True
    if add_layers:
        base_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=2),
        )
    elif not add_layers:
        base_model.fc = nn.Linear(2048, 2)
    return base_model
