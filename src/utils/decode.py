import torch


def _get_peaks(heatmap: torch.Tensor, kernel_size: int = 3):
    """
    Get the peaks from the heatmap.

    Args:
        heatmap (torch.Tensor): The heatmap. (B, C, H, W)
        kernel_size (int): The kernel size for the max pooling.

    Returns:
        peak_ids: (Tuple[torch.Tensor]): The peak locations (x, y). (B, N, 2)
        peak_scores: (Tuple[torch.Tensor]): The peak scores. (B, N)
        peak_classes: (Tuple[torch.Tensor]): The peak classes. (B, N)
    """
    pad = (kernel_size - 1) // 2
    max_pool = torch.nn.functional.max_pool2d(
        heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    peakmap = (heatmap == max_pool).float() * heatmap

    peak_ids, peak_scores, peak_classes = [], [], []

    for batch in peakmap:
        classes, ys, xs = torch.nonzero(batch, as_tuple=True)

        peak_classes.append(classes)
        peak_ids.append(torch.stack([xs, ys], dim=1))
        peak_scores.append(batch[classes, ys, xs])

    return peak_ids, peak_scores, peak_classes


def decode(
    heatmap: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    downsample: float,
    top_k: int = 100
):
    """
    Decode the heatmap, offsets and sizes to the final detections.

    Args:
        heatmap (torch.Tensor): The heatmap. (B, C, H, W)
        offsets (torch.Tensor): The offsets. (B, H, W, 2)
        sizes (torch.Tensor): The sizes. (B, H, W, 2)
        downsample (float): The downsample factor of the feature map.
        top_k (int): The number of detections to keep.

    Returns:
        bboxes: (Tuple[torch.Tensor]): The bounding boxes. (B, N, 4)
        scores: (Tuple[torch.Tensor]): The confident scores. (B, N)
        classes: (Tuple[torch.Tensor]): The class ids. (B, N)
    """
    heatmap = torch.sigmoid_(heatmap)  # transform to range [0, 1]
    peak_ids, peak_scores, peak_classes = _get_peaks(heatmap, kernel_size=3)

    bboxes, scores, classes = [], [], []
    for peak_id, peak_score, peak_class, offset, size in zip(peak_ids, peak_scores, peak_classes, offsets, sizes):
        peak_score, keep_ids = torch.topk(
            peak_score, k=min(top_k, peak_score.size(0)))
        peak_id = peak_id[keep_ids]
        peak_class = peak_class[keep_ids]

        xs, ys = peak_id[:, 0], peak_id[:, 1]

        offset = offset[ys, xs]
        size = size[ys, xs]

        center_xs = (xs + offset[:, 0]) * (1 / downsample)
        center_ys = (ys + offset[:, 1]) * (1 / downsample)
        ws, hs = size[:, 0], size[:, 1]

        xmin = center_xs - ws / 2
        ymin = center_ys - hs / 2
        xmax = center_xs + ws / 2
        ymax = center_ys + hs / 2

        bboxes.append(torch.stack((xmin, ymin, xmax, ymax), dim=1).int())
        scores.append(peak_score)
        classes.append(peak_class)

    return bboxes, scores, classes
