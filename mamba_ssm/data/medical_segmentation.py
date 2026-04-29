from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError:
    nib = None


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
SUPPORTED_SUFFIXES = (*IMAGE_SUFFIXES, '.npy', '.npz', '.nii', '.nii.gz')


def _env_cache_size(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


RAW_ARRAY_CACHE_SIZE = _env_cache_size('MEDSEG_ARRAY_CACHE_SIZE', 64)
SLICE_CACHE_SIZE = _env_cache_size('MEDSEG_SLICE_CACHE_SIZE', 128)


@dataclass(frozen=True)
class PairedCase:
    image_path: Path
    mask_path: Path


def _normalized_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _case_key(path: Path) -> str:
    name = path.name.lower()
    if name.endswith('.nii.gz'):
        return name[:-7]
    return path.stem.lower()


def _is_supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _is_ignored_artifact(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    lowered_name = path.name.lower()
    if any(part == '.ipynb_checkpoints' for part in lowered_parts):
        return True
    if lowered_name.startswith('.'):
        return True
    if '-checkpoint' in lowered_name:
        return True
    return False


def _collect_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f'directory does not exist: {directory}')
    return sorted(
        path for path in directory.rglob('*')
        if path.is_file() and _is_supported(path) and not _is_ignored_artifact(path)
    )


def collect_paired_cases(images_dir: Path, masks_dir: Path) -> List[PairedCase]:
    image_files = _collect_files(Path(images_dir))
    mask_files = _collect_files(Path(masks_dir))
    if not image_files:
        raise FileNotFoundError(f'no supported image files found under {images_dir}')
    if not mask_files:
        raise FileNotFoundError(f'no supported mask files found under {masks_dir}')

    image_map = {}
    for path in image_files:
        key = _case_key(path)
        if key in image_map:
            raise ValueError(f'duplicate image key {key} for files {image_map[key]} and {path}')
        image_map[key] = path

    mask_map = {}
    for path in mask_files:
        key = _case_key(path)
        if key in mask_map:
            raise ValueError(f'duplicate mask key {key} for files {mask_map[key]} and {path}')
        mask_map[key] = path

    common_keys = sorted(set(image_map).intersection(mask_map))
    if not common_keys:
        raise ValueError('no matching image/mask filename pairs were found')

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks:
        raise ValueError(f'missing masks for image ids: {missing_masks[:5]}')
    if missing_images:
        raise ValueError(f'missing images for mask ids: {missing_images[:5]}')

    return [PairedCase(image_path=image_map[key], mask_path=mask_map[key]) for key in common_keys]


@lru_cache(maxsize=RAW_ARRAY_CACHE_SIZE)
def _load_array_cached(path_str: str) -> np.ndarray:
    path = Path(path_str)
    suffix = _normalized_suffix(path)
    if suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
        array = np.asarray(Image.open(path), dtype=np.float32)
    elif suffix == '.npy':
        array = np.load(path, allow_pickle=False)
    elif suffix == '.npz':
        data = np.load(path, allow_pickle=False)
        if not data.files:
            raise ValueError(f'npz file contains no arrays: {path}')
        array = data[data.files[0]]
    elif suffix in {'.nii', '.nii.gz'}:
        if nib is None:
            raise ImportError('nibabel is required for .nii/.nii.gz files. Install it with `pip install nibabel`.')
        array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    else:
        raise ValueError(f'unsupported file type: {path}')
    return np.asarray(array)


@lru_cache(maxsize=SLICE_CACHE_SIZE)
def _load_normalized_slice_cached(
    path_str: str,
    window_min: Optional[float],
    window_max: Optional[float],
) -> np.ndarray:
    image_array = np.asarray(_load_array_cached(path_str))
    return _normalize_image(_to_2d_image(image_array), window_min=window_min, window_max=window_max)


@lru_cache(maxsize=SLICE_CACHE_SIZE)
def _load_binary_mask_cached(path_str: str) -> np.ndarray:
    mask_array = np.asarray(_load_array_cached(path_str))
    return _to_2d_mask(mask_array)


def _pil_resample(name: str):
    resampling = getattr(Image, 'Resampling', Image)
    return getattr(resampling, name)


def _is_2d_image_file(path: Path) -> bool:
    return _normalized_suffix(path) in IMAGE_SUFFIXES


@lru_cache(maxsize=SLICE_CACHE_SIZE)
def _load_resized_normalized_slice_cached(
    path_str: str,
    window_min: Optional[float],
    window_max: Optional[float],
    image_size: int,
) -> np.ndarray:
    path = Path(path_str)
    if _is_2d_image_file(path):
        with Image.open(path) as image:
            image = image.resize((int(image_size), int(image_size)), _pil_resample('BILINEAR'))
            image_array = np.asarray(image, dtype=np.float32)
        return _normalize_image(_to_2d_image(image_array), window_min=window_min, window_max=window_max)

    image_array = np.asarray(_load_array_cached(path_str))
    return _normalize_image(_to_2d_image(image_array), window_min=window_min, window_max=window_max)


@lru_cache(maxsize=SLICE_CACHE_SIZE)
def _load_resized_binary_mask_cached(path_str: str, image_size: int) -> np.ndarray:
    path = Path(path_str)
    if _is_2d_image_file(path):
        with Image.open(path) as mask:
            mask = mask.resize((int(image_size), int(image_size)), _pil_resample('NEAREST'))
            mask_array = np.asarray(mask)
        return _to_2d_mask(mask_array)

    mask_array = np.asarray(_load_array_cached(path_str))
    return _to_2d_mask(mask_array)


def _to_2d_image(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array).squeeze()
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3:
        if array.shape[0] in (1, 3):
            return array.astype(np.float32)
        if array.shape[0] == 4:
            return array[:3].astype(np.float32)
        if array.shape[-1] in (1, 3):
            return np.moveaxis(array, -1, 0).astype(np.float32)
        if array.shape[-1] == 4:
            return np.moveaxis(array[..., :3], -1, 0).astype(np.float32)
    raise ValueError(f'expected a 2D image or squeezeable 2D tensor, got shape {array.shape}')


def _to_2d_mask(array: np.ndarray) -> np.ndarray:
    mask = _to_2d_image(array)
    if mask.ndim == 3:
        # Some 2D segmentation datasets store masks as RGB images where all
        # channels encode the same binary foreground. Collapse them back to a
        # single 2D mask before thresholding.
        mask = mask.max(axis=0)
    return (mask > 0).astype(np.float32)


def _slice_group_info(path: Path) -> Tuple[Optional[str], Optional[int]]:
    match = re.match(r'^(.*)_(\d+)$', path.stem)
    if match is None:
        return None, None
    prefix, index_str = match.groups()
    return f'{path.parent.as_posix()}::{prefix}', int(index_str)


def _take_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    axis = axis if axis >= 0 else volume.ndim + axis
    return np.take(volume, indices=index, axis=axis)


def _normalize_image(image: np.ndarray, window_min: Optional[float], window_max: Optional[float]) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        return np.stack(
            [_normalize_image(channel, window_min=window_min, window_max=window_max) for channel in image],
            axis=0,
        ).astype(np.float32)
    if window_min is not None and window_max is not None:
        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / max(window_max - window_min, 1e-6)
        return image.astype(np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image.astype(np.float32)


def infer_input_mode(cases: Sequence[PairedCase]) -> str:
    if not cases:
        raise ValueError('at least one case is required to infer input mode')
    suffix = _normalized_suffix(cases[0].image_path)
    if suffix in {'.nii', '.nii.gz'}:
        return 'volumes'
    if suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
        return 'slices'
    array = _load_array_cached(str(cases[0].image_path))
    return 'volumes' if np.asarray(array).squeeze().ndim >= 3 else 'slices'


class MedicalSegmentationDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[PairedCase],
        input_mode: str = 'slices',
        image_size: int = 256,
        train: bool = False,
        augment: bool = True,
        slice_axis: int = -1,
        keep_empty_slices: bool = True,
        slice_neighbors: int = 0,
        window_min: Optional[float] = -1200.0,
        window_max: Optional[float] = 400.0,
        random_crop: bool = False,
        crop_prob: float = 0.5,
        crop_scale_min: float = 0.8,
        random_zoom: bool = False,
        zoom_prob: float = 0.5,
        zoom_range: float = 0.15,
    ) -> None:
        super().__init__()
        if input_mode not in {'slices', 'volumes'}:
            raise ValueError("input_mode must be 'slices' or 'volumes'")
        self.cases = list(cases)
        if not self.cases:
            raise ValueError('dataset cannot be empty')
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.train = train
        self.augment = bool(augment)
        self.slice_axis = int(slice_axis)
        self.keep_empty_slices = keep_empty_slices
        self.slice_neighbors = int(slice_neighbors)
        if self.slice_neighbors < 0:
            raise ValueError('slice_neighbors must be >= 0')
        self.window_min = window_min
        self.window_max = window_max
        self.random_crop = bool(random_crop)
        self.crop_prob = float(crop_prob)
        self.crop_scale_min = float(crop_scale_min)
        self.random_zoom = bool(random_zoom)
        self.zoom_prob = float(zoom_prob)
        self.zoom_range = float(zoom_range)
        if not 0.0 <= self.crop_prob <= 1.0:
            raise ValueError('crop_prob must be within [0, 1]')
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError('crop_scale_min must be within (0, 1]')
        if not 0.0 <= self.zoom_prob <= 1.0:
            raise ValueError('zoom_prob must be within [0, 1]')
        if self.zoom_range < 0.0:
            raise ValueError('zoom_range must be >= 0')
        self.samples = self._build_samples()
        self.slice_groups, self.slice_lookup = self._build_slice_groups()
        if not self.samples:
            raise ValueError('dataset contains no usable samples after preprocessing')

    def _build_samples(self) -> List[Tuple[PairedCase, Optional[int]]]:
        if self.input_mode == 'slices':
            samples: List[Tuple[PairedCase, Optional[int]]] = []
            for case in self.cases:
                if not self.keep_empty_slices:
                    mask_array = _load_binary_mask_cached(str(case.mask_path))
                    if mask_array.sum() <= 0:
                        continue
                samples.append((case, None))
            return samples

        samples: List[Tuple[PairedCase, Optional[int]]] = []
        for case in self.cases:
            mask_volume = np.asarray(_load_array_cached(str(case.mask_path))).squeeze()
            if mask_volume.ndim < 3:
                raise ValueError(
                    f'input_mode="volumes" expects 3D masks, got shape {mask_volume.shape} for {case.mask_path}'
                )
            axis = self.slice_axis if self.slice_axis >= 0 else mask_volume.ndim + self.slice_axis
            num_slices = mask_volume.shape[axis]
            for slice_idx in range(num_slices):
                if not self.keep_empty_slices:
                    mask_slice = _take_slice(mask_volume, slice_idx, self.slice_axis)
                    if np.asarray(mask_slice).sum() <= 0:
                        continue
                samples.append((case, slice_idx))
        return samples

    def _build_slice_groups(self):
        if self.input_mode != 'slices' or self.slice_neighbors <= 0:
            return {}, {}
        groups = {}
        lookup = {}
        for case in self.cases:
            group_key, slice_index = _slice_group_info(case.image_path)
            if group_key is None or slice_index is None:
                continue
            group = groups.setdefault(group_key, {})
            group[slice_index] = case.image_path
            lookup[str(case.image_path)] = (group_key, slice_index)
        for group_key, group in list(groups.items()):
            groups[group_key] = {
                'paths': group,
                'indices': sorted(group),
            }
        return groups, lookup

    def _resolve_neighbor_path(self, image_path: Path, offset: int) -> Path:
        if self.slice_neighbors <= 0 or self.input_mode != 'slices':
            return image_path
        lookup = self.slice_lookup.get(str(image_path))
        if lookup is None:
            return image_path
        group_key, center_index = lookup
        group = self.slice_groups.get(group_key)
        if group is None:
            return image_path
        target_index = center_index + offset
        if target_index in group['paths']:
            return group['paths'][target_index]
        indices = group['indices']
        if not indices:
            return image_path
        nearest_index = min(indices, key=lambda value: abs(value - target_index))
        return group['paths'][nearest_index]

    def _load_slice_image(self, image_path: Path) -> np.ndarray:
        return _load_resized_normalized_slice_cached(
            str(image_path),
            self.window_min,
            self.window_max,
            self.image_size,
        )

    def _load_slice_stack(self, image_path: Path) -> np.ndarray:
        if self.slice_neighbors <= 0:
            return self._load_slice_image(image_path)
        slices = []
        for offset in range(-self.slice_neighbors, self.slice_neighbors + 1):
            neighbor_path = self._resolve_neighbor_path(image_path, offset)
            image = self._load_slice_image(neighbor_path)
            if image.ndim == 2:
                slices.append(image)
            else:
                slices.extend(list(image))
        return np.stack(slices, axis=0).astype(np.float32)

    def _apply_random_crop(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.random_crop or self.crop_prob <= 0.0 or random.random() >= self.crop_prob:
            return image_tensor, mask_tensor
        _, height, width = image_tensor.shape
        scale = random.uniform(self.crop_scale_min, 1.0)
        crop_height = max(1, int(round(height * scale)))
        crop_width = max(1, int(round(width * scale)))
        if crop_height >= height and crop_width >= width:
            return image_tensor, mask_tensor
        top = random.randint(0, max(height - crop_height, 0))
        left = random.randint(0, max(width - crop_width, 0))
        image_crop = image_tensor[:, top:top + crop_height, left:left + crop_width].unsqueeze(0)
        mask_crop = mask_tensor[:, top:top + crop_height, left:left + crop_width].unsqueeze(0)
        image_crop = F.interpolate(image_crop, size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
        mask_crop = F.interpolate(mask_crop, size=(height, width), mode='nearest').squeeze(0)
        return image_crop, mask_crop

    def _apply_random_zoom(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.random_zoom or self.zoom_prob <= 0.0 or self.zoom_range <= 0.0 or random.random() >= self.zoom_prob:
            return image_tensor, mask_tensor
        _, height, width = image_tensor.shape
        zoom_factor = random.uniform(max(1e-3, 1.0 - self.zoom_range), 1.0 + self.zoom_range)
        if abs(zoom_factor - 1.0) < 1e-6:
            return image_tensor, mask_tensor
        new_height = max(1, int(round(height * zoom_factor)))
        new_width = max(1, int(round(width * zoom_factor)))
        image_zoom = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        mask_zoom = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode='nearest',
        ).squeeze(0)
        if zoom_factor >= 1.0:
            top = random.randint(0, max(new_height - height, 0))
            left = random.randint(0, max(new_width - width, 0))
            image_zoom = image_zoom[:, top:top + height, left:left + width]
            mask_zoom = mask_zoom[:, top:top + height, left:left + width]
            return image_zoom, mask_zoom

        pad_height = max(height - new_height, 0)
        pad_width = max(width - new_width, 0)
        top = random.randint(0, pad_height)
        left = random.randint(0, pad_width)
        bottom = pad_height - top
        right = pad_width - left
        image_zoom = F.pad(image_zoom, (left, right, top, bottom), mode='constant', value=0.0)
        mask_zoom = F.pad(mask_zoom, (left, right, top, bottom), mode='constant', value=0.0)
        return image_zoom, mask_zoom

    def _apply_resize_and_aug(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image_tensor = torch.from_numpy(image).float()
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 3:
            raise ValueError(f'expected image with shape (H, W) or (C, H, W), got {tuple(image_tensor.shape)}')
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        target_size = (self.image_size, self.image_size)
        if tuple(image_tensor.shape[-2:]) != target_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if tuple(mask_tensor.shape[-2:]) != target_size:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=target_size,
                mode='nearest',
            ).squeeze(0)

        if self.train and self.augment:
            image_tensor, mask_tensor = self._apply_random_crop(image_tensor, mask_tensor)
            image_tensor, mask_tensor = self._apply_random_zoom(image_tensor, mask_tensor)
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            rotations = random.randint(0, 3)
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, rotations, dims=[1, 2])
            if random.random() < 0.5:
                scale = 1.0 + random.uniform(-0.15, 0.15)
                shift = random.uniform(-0.10, 0.10)
                image_tensor = image_tensor * scale + shift
            image_tensor = image_tensor.clamp(0.0, 1.0)

        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case, slice_idx = self.samples[index]
        if self.input_mode == 'volumes':
            mask_array = np.asarray(_load_array_cached(str(case.mask_path)))
            image_array = np.asarray(_load_array_cached(str(case.image_path)))
            image_array = _take_slice(image_array, int(slice_idx), self.slice_axis)
            mask_array = _take_slice(mask_array, int(slice_idx), self.slice_axis)
            image = _normalize_image(_to_2d_image(image_array), self.window_min, self.window_max)
            mask = _to_2d_mask(mask_array)
        else:
            image = self._load_slice_stack(case.image_path)
            mask = _load_resized_binary_mask_cached(str(case.mask_path), self.image_size)
        image_tensor, mask_tensor = self._apply_resize_and_aug(image=image, mask=mask)
        sample_id = case.image_path.stem if slice_idx is None else f'{case.image_path.stem}:{slice_idx}'
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'id': sample_id,
        }
