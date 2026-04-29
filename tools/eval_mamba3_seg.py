from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_ssm.data.fast_binary_segmentation import FastBinarySegmentationDataset
from mamba_ssm.data.medical_segmentation import MedicalSegmentationDataset, collect_paired_cases, infer_input_mode
from tools.train_mamba3_seg import (
    build_model,
    compute_loss,
    load_checkpoint,
    parse_optional_float,
    resolve_amp_dtype,
    resolve_in_channels,
    split_model_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Mamba3 medical segmentation checkpoint on a dataset split and save predictions.'
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to best.pt or last.pt checkpoint.')
    parser.add_argument('--data-root', type=Path, default=None, help='Dataset root containing split/images and split/masks.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split under data-root, for example test or val.')
    parser.add_argument('--images-dir', type=Path, default=None, help='Explicit images directory. Overrides --data-root.')
    parser.add_argument('--masks-dir', type=Path, default=None, help='Explicit masks directory. Overrides --data-root.')
    parser.add_argument('--pred-dir', type=Path, default=Path('outputs/mamba3_seg_eval'), help='Directory for saved predictions and metrics.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true', help='Enable autocast during evaluation.')
    parser.add_argument(
        '--amp-dtype',
        choices=['fp16', 'bf16'],
        default='bf16',
        help='Autocast dtype when --amp is enabled.',
    )
    parser.add_argument('--threshold', type=float, default=0.5, help='Sigmoid threshold for binarizing predictions.')
    parser.add_argument(
        '--tta-mode',
        choices=['none', 'hflip', 'vflip', 'flip'],
        default='none',
        help='Test-time augmentation mode. flip averages identity, horizontal, vertical, and both flips.',
    )
    parser.add_argument('--image-size', type=int, default=None, help='Override image size saved in the checkpoint config.')
    parser.add_argument(
        '--input-mode',
        choices=['auto', 'slices', 'volumes'],
        default=None,
        help='Override input mode saved in the checkpoint config.',
    )
    parser.add_argument('--slice-axis', type=int, default=None, help='Override slice axis saved in the checkpoint config.')
    parser.add_argument('--slice-neighbors', type=int, default=None, help='Override neighboring slices per side for 2.5D input.')
    parser.add_argument('--window-min', type=parse_optional_float, default=None, help='Override window min saved in the checkpoint config.')
    parser.add_argument('--window-max', type=parse_optional_float, default=None, help='Override window max saved in the checkpoint config.')
    parser.add_argument('--save-probs', action='store_true', help='Also save per-sample probability maps as .npy files.')
    return parser.parse_args()


def _checkpoint_value(cli_value, checkpoint_args: Dict, key: str, default=None):
    if cli_value is not None:
        return cli_value
    return checkpoint_args.get(key, default)


def _optional_tuple(value):
    if value is None:
        return None
    return tuple(value)


def _resolve_eval_dirs(args: argparse.Namespace, checkpoint_args: Dict) -> Tuple[Path, Path]:
    if args.images_dir is not None or args.masks_dir is not None:
        if args.images_dir is None or args.masks_dir is None:
            raise ValueError('provide both --images-dir and --masks-dir together')
        return Path(args.images_dir), Path(args.masks_dir)

    if args.data_root is not None:
        data_root = Path(args.data_root)
        return data_root / args.split / 'images', data_root / args.split / 'masks'

    data_root = checkpoint_args.get('data_root')
    if data_root is not None:
        data_root = Path(data_root)
        return data_root / args.split / 'images', data_root / args.split / 'masks'

    if args.split == 'train' and checkpoint_args.get('train_images') and checkpoint_args.get('train_masks'):
        return Path(checkpoint_args['train_images']), Path(checkpoint_args['train_masks'])
    if args.split == 'val' and checkpoint_args.get('val_images') and checkpoint_args.get('val_masks'):
        return Path(checkpoint_args['val_images']), Path(checkpoint_args['val_masks'])

    raise ValueError(
        'could not determine evaluation data directories; provide --data-root or both --images-dir and --masks-dir'
    )


def _sanitize_sample_id(sample_id: str) -> str:
    sample_id = sample_id.replace(':', '__slice_')
    return re.sub(r'[^A-Za-z0-9._-]+', '_', sample_id)


def _save_prediction(path: Path, prediction: torch.Tensor) -> None:
    array = prediction.detach().cpu().squeeze(0).numpy()
    image = Image.fromarray((array * 255.0).astype(np.uint8), mode='L')
    image.save(path)


def _batch_overlap_metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou, intersection, pred_sum, target_sum, union


def _binary_erosion_3x3(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    height, width = mask.shape
    padded = np.pad(mask.astype(bool), 1, mode='constant', constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + height, dx:dx + width]
    return eroded


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = _binary_erosion_3x3(mask)
    boundary = mask & ~eroded
    if not boundary.any():
        return mask
    return boundary


def _pairwise_min_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.size == 0 or dst.size == 0:
        return np.empty((0,), dtype=np.float64)
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    diff = src[:, None, :] - dst[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist.min(axis=1)


def hd95_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if pred_mask.any() != target_mask.any():
        return float('nan')
    pred_boundary = np.argwhere(_mask_boundary(pred_mask))
    target_boundary = np.argwhere(_mask_boundary(target_mask))
    if pred_boundary.size == 0 and target_boundary.size == 0:
        return 0.0
    if pred_boundary.size == 0 or target_boundary.size == 0:
        return float('nan')
    forward = _pairwise_min_distances(pred_boundary, target_boundary)
    backward = _pairwise_min_distances(target_boundary, pred_boundary)
    distances = np.concatenate([forward, backward], axis=0)
    if distances.size == 0:
        return float('nan')
    return float(np.percentile(distances, 95))


def average_precision_from_scores(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = scores.reshape(-1).astype(np.float64)
    targets = targets.reshape(-1).astype(np.uint8)
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind='mergesort')
    targets = targets[order]
    positive_positions = np.flatnonzero(targets == 1)
    if positive_positions.size == 0:
        return 0.0
    precision_at_hits = (np.arange(1, positive_positions.size + 1, dtype=np.float64) / (positive_positions + 1))
    return float(precision_at_hits.mean())


def _tta_flip_dims(mode: str) -> Sequence[Tuple[int, ...]]:
    if mode == 'none':
        return [()]
    if mode == 'hflip':
        return [(), (-1,)]
    if mode == 'vflip':
        return [(), (-2,)]
    if mode == 'flip':
        return [(), (-1,), (-2,), (-2, -1)]
    raise ValueError(f'unsupported tta_mode: {mode}')


def _apply_flip(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    if not dims:
        return x
    return torch.flip(x, dims=list(dims))


def _predict_logits_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    autocast_device: str,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    tta_mode: str,
) -> torch.Tensor:
    logits_accum = None
    flip_dims_list = _tta_flip_dims(tta_mode)
    for flip_dims in flip_dims_list:
        augmented = _apply_flip(images, flip_dims)
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(augmented)
        logits, _ = split_model_outputs(outputs)
        logits = _apply_flip(logits, flip_dims).float()
        logits_accum = logits if logits_accum is None else logits_accum + logits
    assert logits_accum is not None
    return logits_accum / float(len(flip_dims_list))


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location='cpu')
    checkpoint_args = checkpoint.get('args', {})

    model_args = argparse.Namespace(
        in_channels=checkpoint_args.get('in_channels', 1),
        num_classes=checkpoint_args.get('num_classes', 1),
        base_dim=checkpoint_args.get('base_dim', 32),
        depths=checkpoint_args.get('depths', '2,2,2,2,2'),
        d_state=checkpoint_args.get('d_state', 64),
        headdim=checkpoint_args.get('headdim', 32),
        expand=checkpoint_args.get('expand', 2),
        chunk_size=checkpoint_args.get('chunk_size', 64),
        mlp_ratio=checkpoint_args.get('mlp_ratio', 2.0),
        dropout=checkpoint_args.get('dropout', 0.0),
        drop_path=checkpoint_args.get('drop_path', 0.1),
        disable_bidirectional=checkpoint_args.get('disable_bidirectional', False),
        local_mode=checkpoint_args.get('local_mode', 'dmkdc'),
        stage_local_modes=_optional_tuple(checkpoint_args.get('stage_local_modes')),
        mkdc_kernel_sizes=tuple(checkpoint_args.get('mkdc_kernel_sizes', (1, 3, 5))),
        decoder_skip_bridge=checkpoint_args.get('decoder_skip_bridge', 'guided_axial_seq_grn'),
        deep_supervision=checkpoint_args.get('deep_supervision', False),
    )
    if model_args.num_classes != 1:
        raise ValueError(
            f'only binary segmentation checkpoints are currently supported, got num_classes={model_args.num_classes}'
        )

    image_size = int(_checkpoint_value(args.image_size, checkpoint_args, 'image_size', 256))
    slice_axis = int(_checkpoint_value(args.slice_axis, checkpoint_args, 'slice_axis', -1))
    slice_neighbors = int(_checkpoint_value(args.slice_neighbors, checkpoint_args, 'slice_neighbors', 0))
    window_min = _checkpoint_value(args.window_min, checkpoint_args, 'window_min', -1200.0)
    window_max = _checkpoint_value(args.window_max, checkpoint_args, 'window_max', 400.0)

    fast_array = args.data_root is not None and (args.data_root / 'arrays' / f'{args.split}_images.npy').exists()
    if fast_array:
        images_dir = args.data_root / args.split / 'images'
        masks_dir = args.data_root / args.split / 'masks'
        input_mode = 'slices'
        dataset = FastBinarySegmentationDataset(
            args.data_root,
            args.split,
            image_size=image_size,
            train=False,
            augment=False,
        )
    else:
        images_dir, masks_dir = _resolve_eval_dirs(args, checkpoint_args)
        cases = collect_paired_cases(images_dir, masks_dir)
        input_mode = _checkpoint_value(args.input_mode, {'input_mode': checkpoint.get('input_mode')}, 'input_mode', None)
        if input_mode is None:
            input_mode = _checkpoint_value(None, checkpoint_args, 'input_mode', 'auto')
        if input_mode == 'auto':
            input_mode = infer_input_mode(cases)
        dataset = MedicalSegmentationDataset(
            cases,
            input_mode=input_mode,
            image_size=image_size,
            train=False,
            slice_axis=slice_axis,
            keep_empty_slices=True,
            slice_neighbors=slice_neighbors,
            window_min=window_min,
            window_max=window_max,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available, falling back to CPU.')
        device = torch.device('cpu')
    else:
        device = requested_device
    amp_dtype = resolve_amp_dtype(args, device)
    amp_enabled = args.amp and device.type == 'cuda' and amp_dtype is not None
    model_args.in_channels = resolve_in_channels(
        argparse.Namespace(
            in_channels=model_args.in_channels,
            slice_neighbors=slice_neighbors,
        ),
        input_mode,
    )

    model = build_model(model_args).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    bce = nn.BCEWithLogitsLoss()
    pred_dir = args.pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    sample_metrics_path = pred_dir / 'sample_metrics.jsonl'
    metrics_path = pred_dir / 'metrics.json'

    total_loss = 0.0
    total_count = 0
    total_dice = 0.0
    total_iou = 0.0
    global_intersection = 0.0
    global_pred_sum = 0.0
    global_target_sum = 0.0
    global_union = 0.0
    total_ap = 0.0
    total_hd95 = 0.0
    hd95_valid = 0
    all_scores = []
    all_targets = []
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with sample_metrics_path.open('w', encoding='utf-8') as handle:
        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                sample_ids: Sequence[str] = batch['id']

                logits = _predict_logits_with_tta(
                    model=model,
                    images=images,
                    autocast_device=autocast_device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    tta_mode=args.tta_mode,
                )
                loss = compute_loss(
                    logits,
                    masks,
                    bce=bce,
                    loss_mode=checkpoint_args.get('loss_mode', 'dice_bce'),
                    dice_weight=float(checkpoint_args.get('dice_weight', 0.7)),
                    bce_weight=float(checkpoint_args.get('bce_weight', 0.3)),
                    tversky_alpha=float(checkpoint_args.get('tversky_alpha', 0.3)),
                    tversky_beta=float(checkpoint_args.get('tversky_beta', 0.7)),
                    tversky_gamma=float(checkpoint_args.get('tversky_gamma', 0.75)),
                    boundary_loss_weight=float(checkpoint_args.get('boundary_loss_weight', 1.0)),
                    axial_loss_weight=float(checkpoint_args.get('axial_loss_weight', 0.0)),
                    transition_loss_weight=float(checkpoint_args.get('transition_loss_weight', 0.0)),
                )

                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).float()
                dice, iou, intersection, pred_sum, target_sum, union = _batch_overlap_metrics(preds, masks.float())

                batch_size = images.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_dice += float(dice.sum().item())
                total_iou += float(iou.sum().item())
                total_count += batch_size
                global_intersection += float(intersection.sum().item())
                global_pred_sum += float(pred_sum.sum().item())
                global_target_sum += float(target_sum.sum().item())
                global_union += float(union.sum().item())
                all_scores.append(probs.detach().cpu().numpy().reshape(-1).astype(np.float32))
                all_targets.append(masks.detach().cpu().numpy().reshape(-1).astype(np.uint8))

                for idx, sample_id in enumerate(sample_ids):
                    safe_name = _sanitize_sample_id(sample_id)
                    pred_path = pred_dir / f'{safe_name}.png'
                    _save_prediction(pred_path, preds[idx])
                    if args.save_probs:
                        np.save(pred_dir / f'{safe_name}.npy', probs[idx].detach().cpu().squeeze(0).numpy().astype(np.float32))
                    sample_probs = probs[idx].detach().cpu().squeeze(0).numpy()
                    sample_target = masks[idx].detach().cpu().squeeze(0).numpy()
                    sample_pred = preds[idx].detach().cpu().squeeze(0).numpy()
                    sample_ap = average_precision_from_scores(sample_probs, sample_target)
                    sample_hd95 = hd95_from_masks(sample_pred, sample_target)
                    total_ap += sample_ap
                    if np.isfinite(sample_hd95):
                        total_hd95 += sample_hd95
                        hd95_valid += 1
                    handle.write(
                        json.dumps(
                            {
                                'id': sample_id,
                                'dice': float(dice[idx].item()),
                                'iou': float(iou[idx].item()),
                                'ap': float(sample_ap),
                                'hd95': None if not np.isfinite(sample_hd95) else float(sample_hd95),
                                'prediction': str(pred_path),
                            },
                            ensure_ascii=False,
                        )
                        + '\n'
                    )

    global_ap = average_precision_from_scores(
        np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32),
        np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=np.uint8),
    )

    summary = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': int(checkpoint.get('epoch', -1)),
        'checkpoint_best_dice': float(checkpoint.get('best_dice', float('nan'))),
        'images_dir': str(images_dir),
        'masks_dir': str(masks_dir),
        'split': args.split,
        'input_mode': input_mode,
        'image_size': image_size,
        'threshold': float(args.threshold),
        'tta_mode': args.tta_mode,
        'slice_neighbors': int(slice_neighbors),
        'num_samples': int(total_count),
        'mean_loss': total_loss / max(total_count, 1),
        'mean_dice': total_dice / max(total_count, 1),
        'mean_iou': total_iou / max(total_count, 1),
        'mean_ap': total_ap / max(total_count, 1),
        'global_ap': global_ap,
        'mean_hd95': total_hd95 / max(hd95_valid, 1) if hd95_valid > 0 else None,
        'hd95_valid_samples': int(hd95_valid),
        'global_dice': (2.0 * global_intersection + 1e-6) / (global_pred_sum + global_target_sum + 1e-6),
        'global_iou': (global_intersection + 1e-6) / (global_union + 1e-6),
        'pred_dir': str(pred_dir),
        'sample_metrics': str(sample_metrics_path),
    }
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f'Evaluated {total_count} samples from {images_dir}')
    print(
        f'mean_loss={summary["mean_loss"]:.4f} '
        f'mean_dice={summary["mean_dice"]:.4f} '
        f'mean_iou={summary["mean_iou"]:.4f} '
        f'mean_ap={summary["mean_ap"]:.4f} '
        f'global_ap={summary["global_ap"]:.4f} '
        f'mean_hd95={(summary["mean_hd95"] if summary["mean_hd95"] is not None else float("nan")):.4f} '
        f'global_dice={summary["global_dice"]:.4f} '
        f'global_iou={summary["global_iou"]:.4f}'
    )
    print(f'Predictions saved to {pred_dir}')
    print(f'Metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
