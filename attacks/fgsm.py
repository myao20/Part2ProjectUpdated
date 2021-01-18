import torch
from torch import nn


def fgsm(model: nn.Module, images, labels, eps: float, criterion):
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    images.requires_grad = True

    outputs = model(images)

    model.zero_grad()
    y = torch.zeros(list(outputs.size())[0], 2)
    y[range(y.shape[0]), labels] = 1
    y = y.cuda()
    loss = criterion(outputs, y)
    loss.backward()
    perturbed_images = images + eps * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, -1, 1).detach()

    return perturbed_images
