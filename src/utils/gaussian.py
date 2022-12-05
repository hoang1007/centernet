from typing import Tuple
import math
import torch


def gaussian_radius(det_size: Tuple[int, int], min_overlap=0.7):
    """
    Compute the adaptive gaussian radius.
    """
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape: Tuple[int, int], sigma: float = 1):
    """
    Generate a 2D gaussian kernel.

    Args:
        shape (Tuple[int, int]): The shape of the gaussian kernel.
        sigma (float): The standard deviation of the gaussian kernel.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    # y, x = np.ogrid[-m:m + 1, -n:n + 1]
    y = torch.arange(-m, m + 1).view(-1, 1)
    x = torch.arange(-n, n + 1).view(1, -1)

    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap: torch.Tensor, center: Tuple[float, float], radius: float, k: int = 1):
    """
    Draw a 2D gaussian heatmap.

    Args:
        heatmap (torch.Tensor): The heatmap to be drawn. (H, W)
        center (Tuple[float, float]): The center of the gaussian heatmap.
        radius (float): The radius of the gaussian heatmap.
        k (int): The value of the gaussian heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap