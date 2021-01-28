import torch
from torch import nn


def pgd(model: nn.Module, images, labels, eps, criterion, alpha=2/255, iters=40):
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    perturbed_images = images.clone().detach()

    for i in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        if i == 0:
            initial_outputs = outputs

        model.zero_grad()
        y = torch.zeros(list(outputs.size())[0], 2)
        y[range(y.shape[0]), labels] = 1
        y = y.cuda()
        loss = criterion(outputs, y)
        loss.backward()

        perturbed_images = perturbed_images.detach() + alpha * perturbed_images.grad.sign()
        delta = torch.clamp(perturbed_images - images, min=-eps, max=eps)
        images = torch.clamp(images + delta, min=-1, max=1).detach()

    return images.cuda(), initial_outputs
