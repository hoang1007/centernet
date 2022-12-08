from typing import Dict, Optional, Any
from pytorch_lightning import LightningDataModule
from torchvision.datasets import VOCDetection
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


VOC_CLASSES = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_MEAN = (0.485, 0.456, 0.406)
VOC_STD = (0.229, 0.224, 0.225)


class VOCDetectionDataset(Dataset):
    def __init__(self, root: str, year: str = "2007", image_set: str = "trainval"):
        if image_set == "train":
            self.transform = A.Compose(
                (
                    A.Resize(512, 512),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=VOC_MEAN, std=VOC_STD),
                    A.RandomBrightnessContrast(p=0.2),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )
        else:
            self.transform = A.Compose(
                (
                    A.Resize(512, 512),
                    A.Normalize(mean=VOC_MEAN, std=VOC_STD),
                    ToTensorV2(),
                ),
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )

        self.data = VOCDetection(
            root=root, year=year, image_set=image_set, download=True
        )

        self._class2idx = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, info = self.data[idx]
        img = np.array(img)

        gt_boxes, labels = [], []

        for obj_info in info["annotation"]["object"]:
            label_name = obj_info["name"]
            bndbox = [int(k) for k in obj_info["bndbox"].values()]

            gt_boxes.append(bndbox)
            labels.append(self._class2idx[label_name])

        transformed = self.transform(image=img, bboxes=gt_boxes, labels=labels)

        img = transformed["image"]
        gt_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["labels"], dtype=torch.long)

        return img, gt_boxes, labels


class VOCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        year: str = "2007",
        batch_size: int = 64,
        train_ratio: float = 0.75,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.year = year
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_data = None
        self.val_data = None

    def _collate_fn(self, batch):
        imgs, gt_boxes, labels = zip(*batch)

        imgs = torch.stack(imgs, dim=0)

        return imgs, gt_boxes, labels

    def prepare_data(self):
        VOCDetectionDataset(self.data_dir, year=self.year, image_set="trainval")

    def setup(self, stage=None):
        self.train_data = VOCDetectionDataset(
            self.data_dir, year=self.year, image_set="train"
        )
        self.val_data = VOCDetectionDataset(
            self.data_dir, year=self.year, image_set="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
