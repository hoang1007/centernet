import math
from torch import nn


def fill_fc_weights(layers: nn.Module):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_upsample_weights(up: nn.Module):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)

    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))

    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
