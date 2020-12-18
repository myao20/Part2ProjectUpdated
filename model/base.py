import torch.nn as nn
from torchvision import models as models


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
