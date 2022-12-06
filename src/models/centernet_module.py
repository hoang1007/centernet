from typing import Tuple
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torchmetrics import MeanMetric
from utils import draw_umich_gaussian, gaussian_radius


class CenterNet(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gaussian_iou: float = 0.7,
        num_classes: int = 80,
    ):
        super().__init__()

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gaussian_iou = gaussian_iou
        self.num_classes = num_classes

        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, gt_boxes, gt_labels = self._get_inputs(batch)
        heatmap, offset, size = self.net(imgs)

        downsample = heatmap.shape[-1] / imgs.shape[-1]
        keypoints, offsets, sizes = self._get_object_params(
            gt_boxes, downsample=downsample)

        h, w = heatmap.shape[2:]
        gt_heatmaps = self._produce_gt_heatmap(keypoints, gt_labels, self.num_classes, h, w)
        gt_offsets, gt_sizes, masks = self._produce_gt_offset_and_size(keypoints, offsets, sizes, h, w)

        neg_loss = self._compute_neg_loss(heatmap,gt_heatmaps)
        offset_loss = self._compute_reg_loss(offset, gt_offsets, masks)
        size_loss = self._compute_reg_loss(size, gt_sizes, masks)

        loss = neg_loss + offset_loss + size_loss * 0.1

        self.log("train/neg_loss", neg_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train/offset_loss", offset_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train/size_loss", size_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gt_boxes, gt_labels = self._get_inputs(batch)
        heatmap, offset, size = self.net(imgs)

        downsample = heatmap.shape[-1] / imgs.shape[-1]
        keypoints, offsets, sizes = self._get_object_params(
            gt_boxes, downsample=downsample)

        h, w = heatmap.shape[2:]
        gt_heatmaps = self._produce_gt_heatmap(keypoints, gt_labels, self.num_classes, h, w)
        gt_offsets, gt_sizes, masks = self._produce_gt_offset_and_size(keypoints, offsets, sizes, h, w)

        neg_loss = self._compute_neg_loss(heatmap,gt_heatmaps)
        offset_loss = self._compute_reg_loss(offset, gt_offsets, masks)
        size_loss = self._compute_reg_loss(size, gt_sizes, masks)

        loss = neg_loss + offset_loss + size_loss * 0.1

        self.val_loss.update(loss)
        self.log("val/loss", loss, on_step=True)

        return loss

    def validation_epoch_end(self, outputs):
        self.val_loss.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.net.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _compute_neg_loss(self, heatmaps: torch.Tensor, gt_heatmaps: torch.Tensor):
        """
        Compute the modified focal loss for the heatmap.

        Args:
            heatmap (torch.Tensor): The predicted heatmap. (B, C, H, W)
            gt_heatmap (torch.Tensor): The ground truth heatmap. (B, C, H, W)
        """

        pos_ids = gt_heatmaps.eq(1).float()
        neg_ids = gt_heatmaps.lt(1).float()

        neg_weights = torch.pow(1 - gt_heatmaps, 4)

        loss = 0
        for heatmap in heatmaps:
            heatmap = torch.clamp(torch.sigmoid(heatmap),
                                  min=1e-4, max=1 - 1e-4)
            pos_loss = torch.log(heatmap) * torch.pow(1 - heatmap, 2) * pos_ids
            neg_loss = torch.log(1 - heatmap) * \
                torch.pow(heatmap, 2) * neg_weights * neg_ids

            num_pos = pos_ids.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        return loss / len(heatmaps)

    def _compute_reg_loss(self, regs: torch.Tensor, gt_regs: torch.Tensor, masks: torch.BoolTensor):
        """
        Compute the regression loss for the offset and size.

        Args:
            regs (torch.Tensor): The predicted offset and size. (B, 2, H, W)
            gt_regs (torch.Tensor): The ground truth offset and size. (B, 2, H, W)
            masks (torch.BoolTensor): The mask for the offset and size `True` is object else `False`. (B, H, W)
        """
        masks = masks.float().unsqueeze_(1)
        mask_num = masks.float().sum()

        regs = regs * masks
        gt_regs = gt_regs * masks

        loss = nn.functional.l1_loss(regs, gt_regs, reduction='sum')

        return loss / mask_num

    def _get_object_params(
        self,
        gt_boxes: Tuple[torch.Tensor],
        downsample: int
    ):
        """
        Get the keypoints and offsets in the feature map from the ground truth boxes.

        Args:
            gt_boxes (Tuple[torch.Tensor]): The ground truth boxes. (B, N, 4)
            downsample (float): The downsample factor of the feature map.

        Returns:
            keypoints: (Tuple[torch.Tensor]): The center keypoints (x, y) in the feature map. (B, N, 2)
            offsets: (Tuple[torch.Tensor]): The offsets of the keypoints. (B, N, 2)
            sizes: (Tuple[torch.Tensor]): The size of the objects. (B, N, 2)
        """

        keypoints = []
        offsets = []
        sizes = []

        for boxes in gt_boxes:
            centers = downsample * (boxes[:, :2] + boxes[:, 2:]) / 2
            keypoints_ = centers.int()
            offsets_ = centers - keypoints_
            sizes_ = boxes[:, 2:] - boxes[:, :2] + 1
            keypoints.append(keypoints_)
            offsets.append(offsets_)
            sizes.append(sizes_)

        return keypoints, offsets, sizes

    def _produce_gt_heatmap(
        self,
        keypoints: Tuple[torch.Tensor],
        class_ids: torch.Tensor,
        num_classes: int,
        height: int,
        width: int,
    ):
        """
        Produce the ground truth heatmap for the given ground truth boxes.

        Args:
            keypoints (Tuple[torch.Tensor]): The keypoints of objects on feature map. (B, N, 2)
            class_ids (torch.Tensor): The class ids for the ground truth boxes. (B, N)
            num_classes (int): The number of classes. (C)
            height (int): The height of the feature map. (H)
            width (int): The width of the feature map. (W)

        Returns:
            torch.Tensor: The ground truth heatmap. (B, C, H, W)
        """

        gt_heatmaps = torch.zeros((
            len(keypoints),
            num_classes,
            height, width
        ), dtype=torch.float32, device=keypoints[0].device)

        for batch_idx, (kps, cids) in enumerate(zip(keypoints, class_ids)):
            for keypoint, class_id in zip(kps, cids):
                radius = max(
                    0,
                    int(gaussian_radius((height, width),
                        min_overlap=self.gaussian_iou))
                )
                draw_umich_gaussian(gt_heatmaps[batch_idx, class_id], keypoint, radius)

        return gt_heatmaps

    def _produce_gt_offset_and_size(
        self,
        keypoints: Tuple[torch.Tensor],
        offsets: Tuple[torch.Tensor],
        sizes: Tuple[torch.Tensor],
        height: int,
        width: int,
    ):
        """
        Produce the ground truth offsets for the given ground truth boxes.

        Args:
            keypoints (Tuple[torch.Tensor]): The keypoints (x, y) of objects on feature map. (B, N, 2)
            offsets (Tuple[torch.Tensor]): The offsets of the keypoints. (B, N, 2)
            sizes (Tuple[torch.Tensor]): The size of the objects. (B, N, 2)
            height (int): The height of the feature map. (H)
            width (int): The width of the feature map. (W)

        Returns:
            offset_map: (torch.Tensor): The ground truth offsets. (B, 2, H, W)
            size_map: (torch.Tensor): The ground truth sizes. (B, 2, H, W)
            mask: (torch.BoolTensor): The mask for the offset and size `True` is object else `False`. (B, H, W)
        """

        offsets_map = torch.zeros(
            (len(keypoints), height, width, 2), device=keypoints[0].device)
        sizes_map = torch.zeros(
            (len(keypoints), height, width, 2), device=keypoints[0].device)
        masks = torch.zeros((len(keypoints), height, width),
                            device=keypoints[0].device, dtype=torch.bool)

        for batch_idx, (kps, ofs, szs) in enumerate(zip(keypoints, offsets, sizes)):
            x, y = kps[:, 0].long(), kps[:, 1].long()
            offsets_map[batch_idx, y, x] = ofs
            sizes_map[batch_idx, y, x] = szs
            masks[batch_idx, y, x] = True

        offsets_map = offsets_map.moveaxis(3, 1)
        sizes_map = sizes_map.moveaxis(3, 1)

        return offsets_map, sizes_map, masks

    def _get_inputs(self, batch) -> Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        imgs, gt_boxes, labels = batch

        # imgs = imgs.to(self.device)
        # gt_boxes = tuple(boxes.to(self.device) for boxes in gt_boxes)
        # labels = tuple(lbs.to(self.device) for lbs in labels)

        return imgs, gt_boxes, labels