import logging
import torch
from torch import nn

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/cw_linf.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def cw_l_inf(model: nn.Module, images, labels, eps, iters=20, kappa=0):
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    perturbed_images = images.clone().detach()
    alpha = eps/10

    def f(outputs, labels):
        y = torch.zeros(list(outputs.size())[0], 2)
        y[range(y.shape[0]), labels] = 1
        one_hot_labels = y.cuda()
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        return torch.clamp(j - i, min=-kappa)

    for step in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        if step == 0:
            initial_outputs = outputs

        model.zero_grad()
        loss = -f(outputs, labels).sum()

        loss.backward()

        perturbed_images = perturbed_images.detach() + alpha * perturbed_images.grad.sign()
        delta = torch.clamp(perturbed_images - images, min=-eps, max=eps)
        images = torch.clamp(images + delta, min=-1, max=1).detach()

    return images.cuda(), initial_outputs
