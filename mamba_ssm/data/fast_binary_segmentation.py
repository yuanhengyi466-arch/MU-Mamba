from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FastBinarySegmentationDataset(Dataset):
    """Memory-map friendly 2D binary segmentation dataset.

    Expected files:
    `root/arrays/{split}_images.npy`, `root/arrays/{split}_masks.npy`,
    and optionally `root/arrays/{split}_ids.txt`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        image_size: int = 256,
        train: bool = False,
        augment: bool = False,
        random_crop: bool = False,
        crop_prob: float = 0.5,
        crop_scale_min: float = 0.8,
        random_zoom: bool = False,
        zoom_prob: float = 0.5,
        zoom_range: float = 0.15,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.augment = bool(augment)
        self.random_crop = bool(random_crop)
        self.crop_prob = float(crop_prob)
        self.crop_scale_min = float(crop_scale_min)
        self.random_zoom = bool(random_zoom)
        self.zoom_prob = float(zoom_prob)
        self.zoom_range = float(zoom_range)

        array_dir = self.root / "arrays"
        image_path = array_dir / f"{self.split}_images.npy"
        mask_path = array_dir / f"{self.split}_masks.npy"
        ids_path = array_dir / f"{self.split}_ids.txt"
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"missing fast arrays for split={self.split!r} under {array_dir}")

        self.images = np.load(image_path, mmap_mode="r", allow_pickle=False)
        self.masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if len(self.images) != len(self.masks):
            raise ValueError(f"image/mask count mismatch: {len(self.images)} vs {len(self.masks)}")
        if ids_path.exists():
            self.ids = ids_path.read_text(encoding="utf-8").splitlines()
        else:
            self.ids = [f"{self.split}_{idx:05d}" for idx in range(len(self.images))]

    def __len__(self) -> int:
        return int(len(self.images))

    def _resize_if_needed(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = (self.image_size, self.image_size)
        if tuple(image.shape[-2:]) != target:
            image = F.interpolate(image.unsqueeze(0), size=target, mode="bilinear", align_corners=False).squeeze(0)
        if tuple(mask.shape[-2:]) != target:
            mask = F.interpolate(mask.unsqueeze(0), size=target, mode="nearest").squeeze(0)
        return image, mask

    def _apply_random_crop(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.random_crop or self.crop_prob <= 0.0 or random.random() >= self.crop_prob:
            return image, mask
        _, height, width = image.shape
        scale = random.uniform(self.crop_scale_min, 1.0)
        crop_height = max(1, int(round(height * scale)))
        crop_width = max(1, int(round(width * scale)))
        if crop_height >= height and crop_width >= width:
            return image, mask
        top = random.randint(0, max(height - crop_height, 0))
        left = random.randint(0, max(width - crop_width, 0))
        image_crop = image[:, top:top + crop_height, left:left + crop_width].unsqueeze(0)
        mask_crop = mask[:, top:top + crop_height, left:left + crop_width].unsqueeze(0)
        image_crop = F.interpolate(image_crop, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        mask_crop = F.interpolate(mask_crop, size=(height, width), mode="nearest").squeeze(0)
        return image_crop, mask_crop

    def _apply_random_zoom(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.random_zoom or self.zoom_prob <= 0.0 or self.zoom_range <= 0.0 or random.random() >= self.zoom_prob:
            return image, mask
        _, height, width = image.shape
        zoom_factor = random.uniform(max(1e-3, 1.0 - self.zoom_range), 1.0 + self.zoom_range)
        if abs(zoom_factor - 1.0) < 1e-6:
            return image, mask
        new_height = max(1, int(round(height * zoom_factor)))
        new_width = max(1, int(round(width * zoom_factor)))
        image_zoom = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_zoom = F.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode="nearest").squeeze(0)
        if zoom_factor >= 1.0:
            top = random.randint(0, max(new_height - height, 0))
            left = random.randint(0, max(new_width - width, 0))
            return image_zoom[:, top:top + height, left:left + width], mask_zoom[:, top:top + height, left:left + width]

        pad_height = max(height - new_height, 0)
        pad_width = max(width - new_width, 0)
        top = random.randint(0, pad_height)
        left = random.randint(0, pad_width)
        bottom = pad_height - top
        right = pad_width - left
        image_zoom = F.pad(image_zoom, (left, right, top, bottom), mode="constant", value=0.0)
        mask_zoom = F.pad(mask_zoom, (left, right, top, bottom), mode="constant", value=0.0)
        return image_zoom, mask_zoom

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.train or not self.augment:
            return image, mask
        image, mask = self._apply_random_crop(image, mask)
        image, mask = self._apply_random_zoom(image, mask)
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        rotations = random.randint(0, 3)
        if rotations:
            image = torch.rot90(image, rotations, dims=[1, 2])
            mask = torch.rot90(mask, rotations, dims=[1, 2])
        if random.random() < 0.5:
            scale = 1.0 + random.uniform(-0.15, 0.15)
            shift = random.uniform(-0.10, 0.10)
            image = image * scale + shift
        return image.clamp(0.0, 1.0), mask

    def __getitem__(self, index: int):
        image = torch.from_numpy(np.array(self.images[index], dtype=np.float32, copy=True)).float()
        mask = torch.from_numpy(np.array(self.masks[index], dtype=np.float32, copy=True)).float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        image, mask = self._resize_if_needed(image, mask)
        image, mask = self._augment(image, mask)
        mask = (mask > 0.5).float()
        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "id": self.ids[index] if index < len(self.ids) else f"{self.split}_{index:05d}",
        }
