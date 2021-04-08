import logging
import torch
from torch import nn


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('../logs/cw.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def cw_l2(model: nn.Module, images, labels, c=10.0, kappa=0, max_iter=1000, learning_rate=0.01):
    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()
    initial_outputs = model(images)

    # Define f-function
    def f(x):
        outputs = model(x)
        # tensor of one-hot labels
        # one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
        y = torch.zeros(list(outputs.size())[0], 2)
        y[range(y.shape[0]), labels] = 1
        one_hot_labels = y.cuda()
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # wrong logits
        j = torch.masked_select(outputs, one_hot_labels.bool())  # correct logits

        return torch.clamp(j - i, min=-kappa)

    # tensor of zeroes same size as images
    w = torch.zeros_like(images, requires_grad=True).cuda()

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):
        if step % 200 == 0:
            log.debug(f'Step {step} out of {max_iter}')
        a = nn.Tanh()(w)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Stop early if the loss doesn't converge
        if step % (max_iter // 10) == 0:
            if cost > prev:
                log.info('Early stop')
                return a
            prev = cost

    attack_images = nn.Tanh()(w)
    return attack_images.cuda(), initial_outputs
