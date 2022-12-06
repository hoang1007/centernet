import torch
from torchvision.ops import box_iou


def avg_precision(all_pred_bboxes, pred_labels, all_gt_boxes, labels, num_classes, start=1, iou_thresh=0.5):
    precisions = []
    for cls_idx in range(start, num_classes):
        # pred_boxes.shape = (P, 4)
        # gt_boxes.shape = (G, 4)
        pred_boxes = all_pred_bboxes[pred_labels == cls_idx]
        gt_boxes = all_gt_boxes[labels == cls_idx]

        if pred_boxes.size(0) == 0 or gt_boxes.size(0) == 0:
            continue

        # ious.shape = (P, G)
        ious = box_iou(pred_boxes, gt_boxes)

        # iou_argmax.shape = (P,)
        iou_argmax = ious.argmax(dim=1)

        k = ious[
            torch.arange(iou_argmax.size(0), device=iou_argmax.device),
            iou_argmax
        ] > iou_thresh

        prec = (k == 1).sum() / k.size(0)
        precisions.append(prec.item())

    return sum(precisions) / (len(precisions) + 1e-6)
