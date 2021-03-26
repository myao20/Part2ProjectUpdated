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


def cw_l_inf(model: nn.Module, images, labels, eps, alpha=2 / 255, max_iter=1000, kappa=0):
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    adv_images = images.clone().detach()

    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        return torch.clamp(i - j, min=-kappa)

    for step in range(max_iter):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        if step == 0:
            initial_outputs = outputs

        loss = f(outputs, labels).sum()

        if step % 50 == 0:
            log.debug(f'Loss is {loss} after {step} iterations')

        loss.backward()

        # TODO: can try experimenting with changing num iterations
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + eta, min=-1, max=1).detach()

    return adv_images.cuda(), initial_outputs
