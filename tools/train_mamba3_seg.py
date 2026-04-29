from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {'none', 'null'}:
        return None
    return float(value)


def parse_float_list(value: str) -> Tuple[float, ...]:
    scales = tuple(float(part) for part in value.split(',') if part.strip())
    if not scales:
        raise argparse.ArgumentTypeError('expected at least one comma-separated float')
    return scales


def parse_int_list(value: str) -> Tuple[int, ...]:
    items = tuple(int(part) for part in value.split(',') if part.strip())
    if not items:
        raise argparse.ArgumentTypeError('expected at least one comma-separated integer')
    return items


def parse_str_list(value: str) -> Tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(',') if part.strip())
    if not items:
        raise argparse.ArgumentTypeError('expected at least one comma-separated string')
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a Mamba3-based binary medical image segmentation model.')
    parser.add_argument('--data-root', type=Path, default=None, help='Root directory with train/images, train/masks, val/images, val/masks.')
    parser.add_argument('--train-images', type=Path, default=None)
    parser.add_argument('--train-masks', type=Path, default=None)
    parser.add_argument('--val-images', type=Path, default=None)
    parser.add_argument('--val-masks', type=Path, default=None)
    parser.add_argument('--save-dir', type=Path, default=Path('outputs/mamba3_seg'))
    parser.add_argument('--input-mode', choices=['auto', 'slices', 'volumes'], default='auto')
    parser.add_argument('--slice-axis', type=int, default=-1)
    parser.add_argument(
        '--slice-neighbors',
        type=int,
        default=0,
        help='Number of neighboring slices on each side for 2.5D input when using slice data.',
    )
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--keep-empty-slices', action='store_true')
    parser.add_argument('--window-min', type=parse_optional_float, default=-1200.0)
    parser.add_argument('--window-max', type=parse_optional_float, default=400.0)
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--base-dim', type=int, default=32)
    parser.add_argument('--depths', type=str, default='2,2,2,2,2')
    parser.add_argument('--d-state', type=int, default=64)
    parser.add_argument('--headdim', type=int, default=32)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--mlp-ratio', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--disable-bidirectional', action='store_true')
    parser.add_argument(
        '--local-mode',
        choices=['dmkdc'],
        default='dmkdc',
        help='Final MU-Mamba uses DMKDC local feature extraction.',
    )
    parser.add_argument(
        '--stage-local-modes',
        type=parse_str_list,
        default=None,
        help='Optional per-stage local branch modes. The final release uses DMKDC for all stages.',
    )
    parser.add_argument(
        '--mkdc-kernel-sizes',
        type=parse_int_list,
        default=(1, 3, 5),
        help='Comma-separated kernel sizes for the MKDC local branch.',
    )
    parser.add_argument(
        '--decoder-skip-bridge',
        choices=[
            'guided_axial_seq_grn',
        ],
        default='guided_axial_seq_grn',
        help='Final MU-Mamba uses DGASF with GRN skip fusion.',
    )
    parser.add_argument('--deep-supervision', action='store_true', help='Add auxiliary decoder heads during training.')
    parser.add_argument(
        '--aux-weight',
        type=float,
        default=0.2,
        help='Total weight assigned to auxiliary deep-supervision losses.',
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--lr-scheduler-epochs',
        type=int,
        default=None,
        help='Cosine LR schedule length. Defaults to --epochs; set 200 to train 100 epochs with a 200-epoch LR curve.',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=0,
        help='Deprecated no-op kept for CLI compatibility. Training now saves only best.pt.',
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=0,
        help='Stop training if validation Dice does not improve for this many epochs; 0 disables early stopping.',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=0.0, help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--amp-dtype',
        choices=['fp16', 'bf16'],
        default='bf16',
        help='Autocast dtype when --amp is enabled. bf16 is usually more stable on modern NVIDIA GPUs.',
    )
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument(
        '--pretrained-path',
        type=Path,
        default=None,
        help='Optional safe partial pretrained checkpoint loaded before training; optimizer state is not restored.',
    )
    parser.add_argument(
        '--loss-mode',
        choices=[
            'dice_bce',
            'focal_tversky_bce',
            'bce_iou',
            'bce_iou_boundary',
            'bce_iou_axial',
            'bce_iou_axial_transition',
            'bce_iou_transition',
            'bce_iou_guided_transition',
            'bce_iou_guided_transition_align',
        ],
        default='dice_bce',
        help='Segmentation loss recipe.',
    )
    parser.add_argument('--dice-weight', type=float, default=0.7)
    parser.add_argument('--bce-weight', type=float, default=0.3)
    parser.add_argument('--tversky-alpha', type=float, default=0.3, help='False-positive weight for Tversky-based losses.')
    parser.add_argument('--tversky-beta', type=float, default=0.7, help='False-negative weight for Tversky-based losses.')
    parser.add_argument('--tversky-gamma', type=float, default=0.75, help='Focal exponent for focal Tversky loss.')
    parser.add_argument(
        '--boundary-loss-weight',
        type=float,
        default=1.0,
        help='Extra emphasis applied to boundary pixels in bce_iou_boundary mode.',
    )
    parser.add_argument(
        '--axial-loss-weight',
        type=float,
        default=0.2,
        help='Weight of the row/column axial structure loss.',
    )
    parser.add_argument(
        '--transition-loss-weight',
        type=float,
        default=0.1,
        help='Weight of the axial transition consistency loss.',
    )
    parser.add_argument(
        '--gate-align-weight',
        type=float,
        default=0.05,
        help='Weight of the guided axial gate alignment term.',
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='Gradient clipping max-norm. Set <= 0 to disable.',
    )
    parser.add_argument('--no-augment', action='store_true', help='Disable all training-time image augmentations.')
    parser.add_argument('--random-crop', action='store_true', help='Enable random resized crop augmentation during training.')
    parser.add_argument('--crop-prob', type=float, default=0.5, help='Probability of applying random crop when enabled.')
    parser.add_argument('--crop-scale-min', type=float, default=0.8, help='Minimum crop scale relative to the resized image.')
    parser.add_argument('--random-zoom', action='store_true', help='Enable random zoom augmentation during training.')
    parser.add_argument('--zoom-prob', type=float, default=0.5, help='Probability of applying random zoom when enabled.')
    parser.add_argument('--zoom-range', type=float, default=0.15, help='Zoom range sampled from [1-range, 1+range].')
    parser.add_argument(
        '--train-scales',
        type=parse_float_list,
        default=(1.0,),
        help='Comma-separated multi-scale training factors, for example 0.75,1.0,1.25.',
    )
    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_dirs(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    if args.data_root is not None:
        train_images = args.data_root / 'train' / 'images'
        train_masks = args.data_root / 'train' / 'masks'
        val_images = args.data_root / 'val' / 'images'
        val_masks = args.data_root / 'val' / 'masks'
        if val_images.exists() and val_masks.exists():
            return train_images, train_masks, val_images, val_masks
        return train_images, train_masks, None, None

    if args.train_images is None or args.train_masks is None:
        raise ValueError('provide either --data-root or both --train-images and --train-masks')
    return args.train_images, args.train_masks, args.val_images, args.val_masks


def build_datasets(args: argparse.Namespace):
    if args.data_root is not None and (args.data_root / 'arrays' / 'train_images.npy').exists():
        from mamba_ssm.data.fast_binary_segmentation import FastBinarySegmentationDataset

        val_array = args.data_root / 'arrays' / 'val_images.npy'
        if not val_array.exists():
            raise FileNotFoundError(f'missing fast validation arrays under {args.data_root / "arrays"}')
        train_dataset = FastBinarySegmentationDataset(
            args.data_root,
            'train',
            image_size=args.image_size,
            train=True,
            augment=not args.no_augment,
            random_crop=args.random_crop,
            crop_prob=args.crop_prob,
            crop_scale_min=args.crop_scale_min,
            random_zoom=args.random_zoom,
            zoom_prob=args.zoom_prob,
            zoom_range=args.zoom_range,
        )
        val_dataset = FastBinarySegmentationDataset(
            args.data_root,
            'val',
            image_size=args.image_size,
            train=False,
            augment=False,
        )
        return train_dataset, val_dataset, 'slices'

    from mamba_ssm.data.medical_segmentation import (
        MedicalSegmentationDataset,
        collect_paired_cases,
        infer_input_mode,
    )

    train_images, train_masks, val_images, val_masks = resolve_data_dirs(args)
    train_cases = collect_paired_cases(train_images, train_masks)
    if val_images is None or val_masks is None:
        raise FileNotFoundError(
            'validation folders are required: provide data-root/val/images and data-root/val/masks, '
            'or pass --val-images and --val-masks explicitly.'
        )
    val_cases = collect_paired_cases(val_images, val_masks)

    input_mode = args.input_mode
    if input_mode == 'auto':
        input_mode = infer_input_mode(train_cases)

    train_dataset = MedicalSegmentationDataset(
        train_cases,
        input_mode=input_mode,
        image_size=args.image_size,
        train=True,
        augment=not args.no_augment,
        slice_axis=args.slice_axis,
        keep_empty_slices=args.keep_empty_slices,
        slice_neighbors=args.slice_neighbors,
        window_min=args.window_min,
        window_max=args.window_max,
        random_crop=args.random_crop,
        crop_prob=args.crop_prob,
        crop_scale_min=args.crop_scale_min,
        random_zoom=args.random_zoom,
        zoom_prob=args.zoom_prob,
        zoom_range=args.zoom_range,
    )
    val_dataset = MedicalSegmentationDataset(
        val_cases,
        input_mode=input_mode,
        image_size=args.image_size,
        train=False,
        augment=False,
        slice_axis=args.slice_axis,
        keep_empty_slices=True,
        slice_neighbors=args.slice_neighbors,
        window_min=args.window_min,
        window_max=args.window_max,
        random_crop=False,
        crop_prob=0.0,
        crop_scale_min=args.crop_scale_min,
        random_zoom=False,
        zoom_prob=0.0,
        zoom_range=args.zoom_range,
    )
    return train_dataset, val_dataset, input_mode


def build_model(args: argparse.Namespace) -> nn.Module:
    try:
        from mamba_ssm.models.vision_mamba3_seg import VisionMamba3Seg
    except ImportError as exc:
        raise RuntimeError(
            'Failed to import the Mamba3 segmentation model. '
            'Install the repo dependencies first, for example `py -3 -m pip install -e .`, '
            'and then install `requirements-medseg.txt` if needed.'
        ) from exc

    depths = tuple(int(part) for part in args.depths.split(',') if part.strip())
    dims = tuple(args.base_dim * (2 ** idx) for idx in range(len(depths)))
    model = VisionMamba3Seg(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        dims=dims,
        depths=depths,
        d_state=args.d_state,
        headdim=args.headdim,
        expand=args.expand,
        chunk_size=args.chunk_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        drop_path_rate=args.drop_path,
        bidirectional=not args.disable_bidirectional,
        local_mode=args.local_mode,
        stage_local_modes=args.stage_local_modes,
        mkdc_kernel_sizes=args.mkdc_kernel_sizes,
        decoder_skip_bridge=args.decoder_skip_bridge,
        mamba_branch_mode='default',
        deep_supervision=args.deep_supervision,
    )
    return model


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': drop_last,
    }
    if num_workers > 0:
        # Reuse worker processes across epochs to avoid repeated startup costs.
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, **loader_kwargs)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def dice_score_tensor_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    return float(dice_score_tensor_from_logits(logits, targets, threshold=threshold, eps=eps).item())


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    true_pos = (probs * targets).sum(dim=dims)
    false_pos = (probs * (1.0 - targets)).sum(dim=dims)
    false_neg = ((1.0 - probs) * targets).sum(dim=dims)
    tversky = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)
    return torch.pow(1.0 - tversky, gamma).mean()


def iou_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def boundary_band(targets: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp_(0.0, 1.0)


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1e-6)


def weighted_iou_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    weighted_intersection = (probs * targets * weights).sum(dim=dims)
    weighted_union = ((probs + targets - probs * targets) * weights).sum(dim=dims)
    iou = (weighted_intersection + eps) / (weighted_union + eps)
    return 1.0 - iou.mean()


def sequence_dice_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def axial_structure_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_row = probs.amax(dim=-1)
    target_row = targets.amax(dim=-1)
    pred_col = probs.amax(dim=-2)
    target_col = targets.amax(dim=-2)
    row_loss = sequence_dice_loss(pred_row, target_row, eps=eps)
    col_loss = sequence_dice_loss(pred_col, target_col, eps=eps)
    return 0.5 * (row_loss + col_loss)


def axial_transition_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    target_dh = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
    pred_dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
    target_dw = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
    loss_h = F.smooth_l1_loss(pred_dh, target_dh)
    loss_w = F.smooth_l1_loss(pred_dw, target_dw)
    return 0.5 * (loss_h + loss_w)


def _transition_maps(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    probs = torch.sigmoid(logits)
    pred_dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
    target_dh = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
    pred_dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
    target_dw = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
    return pred_dh, target_dh, pred_dw, target_dw


def guided_axial_transition_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gate_maps: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not gate_maps:
        return axial_transition_loss(logits, targets)
    pred_dh, target_dh, pred_dw, target_dw = _transition_maps(logits, targets)
    height, width = targets.shape[-2:]
    losses = []
    for gates in gate_maps:
        gates = F.interpolate(gates, size=(height, width), mode='bilinear', align_corners=False)
        row_gate = gates[:, 0:1]
        col_gate = gates[:, 1:2]
        # Row-wise sequence gates supervise horizontal transitions; column-wise gates supervise vertical transitions.
        weight_dw = 1.0 + 0.5 * (row_gate[:, :, :, 1:] + row_gate[:, :, :, :-1])
        weight_dh = 1.0 + 0.5 * (col_gate[:, :, 1:, :] + col_gate[:, :, :-1, :])
        loss_w = F.smooth_l1_loss(pred_dw, target_dw, reduction='none')
        loss_h = F.smooth_l1_loss(pred_dh, target_dh, reduction='none')
        loss_w = (loss_w * weight_dw).sum() / weight_dw.sum().clamp_min(1e-6)
        loss_h = (loss_h * weight_dh).sum() / weight_dh.sum().clamp_min(1e-6)
        losses.append(0.5 * (loss_h + loss_w))
    return torch.stack(losses).mean()


def guided_axial_gate_alignment_loss(
    targets: torch.Tensor,
    gate_maps: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not gate_maps:
        return targets.new_tensor(0.0)
    target_dh = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :]).clamp(0.0, 1.0)
    target_dw = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1]).clamp(0.0, 1.0)
    height, width = targets.shape[-2:]
    losses = []
    for gates in gate_maps:
        gates = F.interpolate(gates, size=(height, width), mode='bilinear', align_corners=False)
        row_gate = 0.5 * (gates[:, 0:1, :, 1:] + gates[:, 0:1, :, :-1])
        col_gate = 0.5 * (gates[:, 1:2, 1:, :] + gates[:, 1:2, :-1, :])
        row_logits = torch.logit(row_gate.float().clamp(1e-4, 1.0 - 1e-4))
        col_logits = torch.logit(col_gate.float().clamp(1e-4, 1.0 - 1e-4))
        loss_w = F.binary_cross_entropy_with_logits(row_logits, target_dw.float())
        loss_h = F.binary_cross_entropy_with_logits(col_logits, target_dh.float())
        losses.append(0.5 * (loss_h + loss_w))
    return torch.stack(losses).mean()


def apply_train_scale(
    images: torch.Tensor,
    masks: torch.Tensor,
    train_scales: Tuple[float, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(train_scales) <= 1:
        return images, masks
    scale = random.choice(train_scales)
    if abs(scale - 1.0) < 1e-6:
        return images, masks
    _, _, height, width = images.shape
    scaled_height = max(32, int(round(height * scale / 32.0)) * 32)
    scaled_width = max(32, int(round(width * scale / 32.0)) * 32)
    if scaled_height == height and scaled_width == width:
        return images, masks
    images = torch.nn.functional.interpolate(
        images,
        size=(scaled_height, scaled_width),
        mode='bilinear',
        align_corners=False,
    )
    masks = torch.nn.functional.interpolate(
        masks,
        size=(scaled_height, scaled_width),
        mode='nearest',
    )
    return images, masks


def split_model_outputs(outputs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if isinstance(outputs, tuple):
        main_logits = outputs[0]
        aux_outputs = outputs[1] if len(outputs) > 1 else ()
    elif isinstance(outputs, list):
        main_logits = outputs[0]
        aux_outputs = outputs[1:]
    else:
        return outputs, ()
    if isinstance(aux_outputs, torch.Tensor):
        aux_outputs = (aux_outputs,)
    else:
        aux_outputs = tuple(aux_outputs)
    return main_logits, aux_outputs


def compute_total_loss(
    logits,
    targets: torch.Tensor,
    bce: nn.Module,
    loss_mode: str,
    dice_weight: float,
    bce_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    boundary_loss_weight: float,
    aux_weight: float = 0.0,
    axial_loss_weight: float = 0.0,
    transition_loss_weight: float = 0.0,
    gate_align_weight: float = 0.0,
    guided_axial_gates: Sequence[torch.Tensor] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    main_logits, aux_logits = split_model_outputs(logits)
    loss = compute_loss(
        main_logits,
        targets,
        bce=bce,
        loss_mode=loss_mode,
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        tversky_gamma=tversky_gamma,
        boundary_loss_weight=boundary_loss_weight,
        axial_loss_weight=axial_loss_weight,
        transition_loss_weight=transition_loss_weight,
    )
    if aux_logits and aux_weight > 0:
        per_aux_weight = aux_weight / float(len(aux_logits))
        for aux in aux_logits:
            loss = loss + per_aux_weight * compute_loss(
                aux,
                targets,
                bce=bce,
                loss_mode=loss_mode,
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                boundary_loss_weight=boundary_loss_weight,
                axial_loss_weight=axial_loss_weight,
                transition_loss_weight=transition_loss_weight,
            )
    if loss_mode in {'bce_iou_guided_transition', 'bce_iou_guided_transition_align'}:
        loss = loss + float(transition_loss_weight) * guided_axial_transition_loss(
            main_logits,
            targets,
            guided_axial_gates,
        )
        if loss_mode == 'bce_iou_guided_transition_align':
            loss = loss + float(gate_align_weight) * guided_axial_gate_alignment_loss(
                targets,
                guided_axial_gates,
            )
    return loss, main_logits


def needs_guided_axial_gates(loss_mode: str, transition_loss_weight: float, gate_align_weight: float) -> bool:
    if loss_mode == 'bce_iou_guided_transition':
        return float(transition_loss_weight) > 0
    if loss_mode == 'bce_iou_guided_transition_align':
        return float(transition_loss_weight) > 0 or float(gate_align_weight) > 0
    return False


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce: nn.Module,
    loss_mode: str,
    dice_weight: float,
    bce_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    boundary_loss_weight: float,
    axial_loss_weight: float = 0.0,
    transition_loss_weight: float = 0.0,
) -> torch.Tensor:
    if loss_mode == 'dice_bce':
        main_loss = dice_loss(logits, targets)
        aux_bce = bce(logits, targets)
    elif loss_mode == 'focal_tversky_bce':
        main_loss = focal_tversky_loss(
            logits,
            targets,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )
        aux_bce = bce(logits, targets)
    elif loss_mode == 'bce_iou':
        main_loss = iou_loss(logits, targets)
        aux_bce = bce(logits, targets)
    elif loss_mode == 'bce_iou_boundary':
        weights = 1.0 + float(boundary_loss_weight) * boundary_band(targets)
        main_loss = weighted_iou_loss(logits, targets, weights)
        aux_bce = weighted_bce_loss(logits, targets, weights)
    elif loss_mode == 'bce_iou_axial':
        main_loss = iou_loss(logits, targets)
        aux_bce = bce(logits, targets)
        return (
            dice_weight * main_loss
            + bce_weight * aux_bce
            + float(axial_loss_weight) * axial_structure_loss(logits, targets)
        )
    elif loss_mode == 'bce_iou_axial_transition':
        main_loss = iou_loss(logits, targets)
        aux_bce = bce(logits, targets)
        return (
            dice_weight * main_loss
            + bce_weight * aux_bce
            + float(axial_loss_weight) * axial_structure_loss(logits, targets)
            + float(transition_loss_weight) * axial_transition_loss(logits, targets)
        )
    elif loss_mode == 'bce_iou_transition':
        main_loss = iou_loss(logits, targets)
        aux_bce = bce(logits, targets)
        return (
            dice_weight * main_loss
            + bce_weight * aux_bce
            + float(transition_loss_weight) * axial_transition_loss(logits, targets)
        )
    elif loss_mode in {'bce_iou_guided_transition', 'bce_iou_guided_transition_align'}:
        main_loss = iou_loss(logits, targets)
        aux_bce = bce(logits, targets)
    else:
        raise ValueError(f'unsupported loss_mode: {loss_mode}')
    return dice_weight * main_loss + bce_weight * aux_bce


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> Optional[torch.dtype]:
    if not args.amp or device.type != 'cuda':
        return None
    if args.amp_dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print('CUDA bf16 is not supported on this device, falling back to fp16 AMP.')
    return torch.float16


def configure_runtime(device: torch.device) -> None:
    # Favor faster matmul kernels when training tolerates small FP32 precision tradeoffs.
    try:
        torch.set_float32_matmul_precision('high')
    except (AttributeError, RuntimeError):
        pass
    if device.type != 'cuda':
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def create_grad_scaler(device: torch.device, enabled: bool):
    if device.type != 'cuda':
        enabled = False
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_in_channels(args: argparse.Namespace, input_mode: str) -> int:
    if args.slice_neighbors <= 0:
        return int(args.in_channels)
    if input_mode != 'slices':
        raise ValueError('--slice-neighbors is only supported when input_mode resolves to "slices"')
    factor = 2 * int(args.slice_neighbors) + 1
    current_in_channels = int(args.in_channels)
    if current_in_channels in (1, 3):
        base_channels = current_in_channels
        expected_in_channels = base_channels * factor
        return expected_in_channels
    if current_in_channels % factor == 0 and current_in_channels // factor in (1, 3):
        return current_in_channels
    expected_values = [base * factor for base in (1, 3)]
    raise ValueError(
        f'--slice-neighbors {args.slice_neighbors} expects --in-channels to be 1, 3, {expected_values[0]}, or {expected_values[1]}, '
        f'got {args.in_channels}'
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    bce: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    loss_mode: str,
    dice_weight: float,
    bce_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    boundary_loss_weight: float,
    grad_clip: float,
    train_scales: Tuple[float, ...],
    aux_weight: float,
    axial_loss_weight: float = 0.0,
    transition_loss_weight: float = 0.0,
    gate_align_weight: float = 0.0,
) -> Dict[str, float]:
    model.train()
    running_loss = torch.zeros((), device=device)
    running_dice = torch.zeros((), device=device)
    total = 0
    skipped = 0
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        images, masks = apply_train_scale(images, masks, train_scales)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(images)
            guided_axial_gates = (
                model.guided_axial_gates()
                if needs_guided_axial_gates(loss_mode, transition_loss_weight, gate_align_weight)
                and hasattr(model, 'guided_axial_gates')
                else ()
            )
            loss, logits = compute_total_loss(
                outputs,
                masks,
                bce=bce,
                loss_mode=loss_mode,
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                boundary_loss_weight=boundary_loss_weight,
                aux_weight=aux_weight,
                axial_loss_weight=axial_loss_weight,
                transition_loss_weight=transition_loss_weight,
                gate_align_weight=gate_align_weight,
                guided_axial_gates=guided_axial_gates,
            )
        if not torch.isfinite(loss):
            skipped += 1
            print(f'Warning: non-finite training loss encountered for batch {skipped}; skipping optimizer step.')
            continue
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    skipped += 1
                    print(f'Warning: non-finite gradient norm encountered ({float(grad_norm):.4f}); skipping optimizer step.')
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    skipped += 1
                    print(f'Warning: non-finite gradient norm encountered ({float(grad_norm):.4f}); skipping optimizer step.')
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
        batch_size = images.shape[0]
        running_loss += loss.detach() * batch_size
        running_dice += dice_score_tensor_from_logits(logits.detach(), masks) * batch_size
        total += batch_size
    return {
        'loss': float((running_loss / max(total, 1)).item()),
        'dice': float((running_dice / max(total, 1)).item()),
        'skipped': float(skipped),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    loss_mode: str,
    dice_weight: float,
    bce_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    boundary_loss_weight: float,
    aux_weight: float,
    axial_loss_weight: float = 0.0,
    transition_loss_weight: float = 0.0,
    gate_align_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    running_loss = torch.zeros((), device=device)
    running_dice = torch.zeros((), device=device)
    total = 0
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(images)
            guided_axial_gates = (
                model.guided_axial_gates()
                if needs_guided_axial_gates(loss_mode, transition_loss_weight, gate_align_weight)
                and hasattr(model, 'guided_axial_gates')
                else ()
            )
            loss, logits = compute_total_loss(
                outputs,
                masks,
                bce=bce,
                loss_mode=loss_mode,
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                boundary_loss_weight=boundary_loss_weight,
                aux_weight=aux_weight if model.training else 0.0,
                axial_loss_weight=axial_loss_weight,
                transition_loss_weight=transition_loss_weight,
                gate_align_weight=gate_align_weight,
                guided_axial_gates=guided_axial_gates,
            )
        if not torch.isfinite(loss):
            return {'loss': float('nan'), 'dice': 0.0}
        batch_size = images.shape[0]
        running_loss += loss.detach() * batch_size
        running_dice += dice_score_tensor_from_logits(logits, masks) * batch_size
        total += batch_size
    return {
        'loss': float((running_loss / max(total, 1)).item()),
        'dice': float((running_dice / max(total, 1)).item()),
    }


def save_checkpoint(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = 'cpu') -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main() -> None:
    args = parse_args()
    if args.save_every < 0:
        raise ValueError('--save-every must be >= 0')
    if args.early_stop_patience < 0:
        raise ValueError('--early-stop-patience must be >= 0')
    if args.aux_weight < 0:
        raise ValueError('--aux-weight must be >= 0')
    if args.boundary_loss_weight < 0:
        raise ValueError('--boundary-loss-weight must be >= 0')
    if args.axial_loss_weight < 0:
        raise ValueError('--axial-loss-weight must be >= 0')
    if args.transition_loss_weight < 0:
        raise ValueError('--transition-loss-weight must be >= 0')
    if args.gate_align_weight < 0:
        raise ValueError('--gate-align-weight must be >= 0')
    if args.save_every > 0:
        print('Ignoring --save-every: periodic checkpoints are disabled; only best.pt will be saved.')
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, input_mode = build_datasets(args)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available, falling back to CPU. Note: Mamba3 training is intended for CUDA.')
        device = torch.device('cpu')
    else:
        device = requested_device
    configure_runtime(device)
    amp_dtype = resolve_amp_dtype(args, device)
    amp_enabled = args.amp and device.type == 'cuda' and amp_dtype is not None
    args.in_channels = resolve_in_channels(args, input_mode)

    model = build_model(args).to(device)
    if args.pretrained_path is not None:
        from tools.pretrained_loading import load_partial_pretrained

        load_partial_pretrained(model, args.pretrained_path, label='binary segmentation pretrained')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler_epochs = max(int(args.lr_scheduler_epochs or args.epochs), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr_scheduler_epochs,
        eta_min=args.min_lr,
    )
    scaler = create_grad_scaler(device=device, enabled=amp_enabled and amp_dtype == torch.float16)
    bce = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_dice = -math.inf
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_dice = float(checkpoint.get('best_dice', best_dice))
        epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', epochs_without_improvement))
        print(f'Resumed from {args.resume} at epoch {start_epoch}.')

    config_path = args.save_dir / 'train_config.json'
    with config_path.open('w', encoding='utf-8') as handle:
        json.dump({**vars(args), 'input_mode': input_mode}, handle, indent=2, default=str)

    print(f'Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)} | Input mode: {input_mode}')
    if amp_enabled:
        print(f'AMP enabled with dtype={args.amp_dtype}.')
    if lr_scheduler_epochs != int(args.epochs):
        print(f'LR scheduler uses {lr_scheduler_epochs} epochs while training runs for {args.epochs} epochs.')
    epoch_digits = max(3, len(str(args.epochs)))
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            bce=bce,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            loss_mode=args.loss_mode,
            dice_weight=args.dice_weight,
            bce_weight=args.bce_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            boundary_loss_weight=args.boundary_loss_weight,
            grad_clip=args.grad_clip,
            train_scales=args.train_scales,
            aux_weight=args.aux_weight,
            axial_loss_weight=args.axial_loss_weight,
            transition_loss_weight=args.transition_loss_weight,
            gate_align_weight=args.gate_align_weight,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            bce=bce,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            loss_mode=args.loss_mode,
            dice_weight=args.dice_weight,
            bce_weight=args.bce_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            boundary_loss_weight=args.boundary_loss_weight,
            aux_weight=0.0,
            axial_loss_weight=args.axial_loss_weight,
            transition_loss_weight=args.transition_loss_weight,
            gate_align_weight=args.gate_align_weight,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1:03d}/{args.epochs:03d} '
            f'lr={current_lr:.6f} '
            f'train_loss={train_metrics["loss"]:.4f} '
            f'train_dice={train_metrics["dice"]:.4f} '
            f'val_loss={val_metrics["loss"]:.4f} '
            f'val_dice={val_metrics["dice"]:.4f} '
            f'skipped={int(train_metrics["skipped"])}'
        )

        payload = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_dice': max(best_dice, val_metrics['dice']),
            'epochs_without_improvement': epochs_without_improvement,
            'args': vars(args),
            'input_mode': input_mode,
        }
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            epochs_without_improvement = 0
            payload['best_dice'] = best_dice
            payload['epochs_without_improvement'] = epochs_without_improvement
            save_checkpoint(args.save_dir / 'best.pt', payload)
            print(f'New best checkpoint saved to {args.save_dir / "best.pt"}')
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f'Early stopping: no validation Dice improvement for {args.early_stop_patience} epochs.')
            break

    print(f'Training finished. Best validation Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
