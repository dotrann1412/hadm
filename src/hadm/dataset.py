"""
HAD (Human Artifact Dataset) loader.

Reads the per-image JSON annotation format from the HAD dataset
(Wang et al., "Detecting Human Artifacts from Text-to-Image Models").

Directory layout expected:
    root/
    ├── images/
    │   ├── train_ALL/
    │   ├── val_ALL/
    │   ├── val_sdxl/
    │   └── ...
    └── annotations/
        ├── train_ALL/
        ├── val_ALL/
        └── ...
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

LOCAL_CLASSES = ["face", "torso", "arm", "leg", "hand", "feet"]
GLOBAL_CLASSES = [
    "human missing arm", "human missing face", "human missing feet",
    "human missing hand", "human missing leg", "human missing torso",
    "human with extra arm", "human with extra face", "human with extra feet",
    "human with extra hand", "human with extra leg", "human with extra torso",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class HADDataset(Dataset):
    """Human Artifact Dataset for training / evaluating HADM detectors.

    Each image has a matching JSON annotation file.  For the *local* variant the
    JSON contains ``annotation`` entries with ``body_parts`` + ``bbox``.  For the
    *global* variant it contains ``human`` entries with ``bbox`` + ``tag`` list.

    Real-image negative samples (no artifacts) are supported — they simply have
    empty annotation lists, which means zero target boxes.

    Args:
        root: Path to the ``human_artifact_dataset`` directory.
        split: Sub-folder name under ``images/`` and ``annotations/``
               (e.g. ``train_ALL``, ``val_ALL``, ``val_sdxl``).
        mode: ``'local'`` for 6 body-part classes, ``'global'`` for 12 anomaly classes.
        train: If True, apply data augmentation (color jitter + random H-flip).
        real_negative_dirs: Optional list of additional image directories that contain
            only real (non-generated) images with no annotations. These serve as
            hard negatives so the model sees artifact-free humans during training.
    """

    def __init__(
        self,
        root: str,
        split: str = "train_ALL",
        mode: str = "local",
        train: bool = False,
        real_negative_dirs: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.mode = mode
        self.train = train

        self.img_dir = self.root / "images" / split
        self.anno_dir = self.root / "annotations" / split

        if mode == "local":
            self.class_to_idx = {c: i for i, c in enumerate(LOCAL_CLASSES)}
        else:
            self.class_to_idx = {c: i for i, c in enumerate(GLOBAL_CLASSES)}

        self.samples: list[tuple[Path, Optional[Path]]] = []

        if self.img_dir.is_dir():
            for fname in sorted(os.listdir(self.img_dir)):
                if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                img_path = self.img_dir / fname
                anno_name = Path(fname).stem + ".json"
                anno_path = self.anno_dir / anno_name
                self.samples.append(
                    (img_path, anno_path if anno_path.exists() else None)
                )

        if real_negative_dirs:
            for neg_dir in real_negative_dirs:
                neg_path = Path(neg_dir)
                if not neg_path.is_dir():
                    continue
                for fname in sorted(os.listdir(neg_path)):
                    if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    self.samples.append((neg_path / fname, None))

        if train:
            self.color_jitter = T.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5
            )
        else:
            self.color_jitter = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, anno_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        boxes, labels = self._parse_annotations(anno_path)

        if self.train:
            image, boxes = self._augment(image, boxes)

        image = TF.to_tensor(image)  # [C, H, W] float32 in [0, 1]

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64) + 1  # 0 = background
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": idx,
            "area": area,
            "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
        }
        return image, target

    # ------------------------------------------------------------------
    # Annotation parsing
    # ------------------------------------------------------------------

    def _parse_annotations(self, anno_path: Optional[Path]):
        boxes, labels = [], []
        if anno_path is None:
            return boxes, labels

        with open(anno_path) as f:
            anno = json.load(f)

        if self.mode == "local":
            for entry in anno.get("annotation", []):
                part = entry.get("body_parts", "")
                if part not in self.class_to_idx:
                    continue
                bbox = self._sanitize_bbox(entry["bbox"])
                if bbox is None:
                    continue
                boxes.append(bbox)
                labels.append(self.class_to_idx[part])
        else:
            for human in anno.get("human", []):
                bbox = self._sanitize_bbox(human["bbox"])
                if bbox is None:
                    continue
                for tag in human.get("tag", []):
                    if tag not in self.class_to_idx:
                        continue
                    boxes.append(bbox)
                    labels.append(self.class_to_idx[tag])

        return boxes, labels

    @staticmethod
    def _sanitize_bbox(bbox: list[float]) -> Optional[list[float]]:
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return [x1, y1, x2, y2]

    # ------------------------------------------------------------------
    # Data augmentation (training only)
    # ------------------------------------------------------------------

    def _augment(self, image: Image.Image, boxes: list[list[float]]):
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        if random.random() > 0.5:
            image = TF.hflip(image)
            w = image.width
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

        return image, boxes


class ImageFolderDataset(Dataset):
    """Simple loader for inference on a folder of images (no annotations)."""

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)
        return image, str(img_path)


def collate_fn(batch):
    """Custom collate for variable-size images in detection."""
    return tuple(zip(*batch))
