# Binary Segmentation Experiment Settings

This file records the recommended binary segmentation setup for MU-Mamba.

## Dataset Layout

Each dataset root should contain:

```text
train/images
train/masks
val/images
val/masks
test/images
test/masks
```

Fast datasets additionally contain:

```text
arrays/train_images.npy
arrays/train_masks.npy
arrays/train_ids.txt
arrays/val_images.npy
arrays/val_masks.npy
arrays/val_ids.txt
arrays/test_images.npy
arrays/test_masks.npy
arrays/test_ids.txt
```

## Default Binary Setup

| Item | Value |
|---|---|
| Image size | 256 x 256 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Epochs | 200 |
| Batch size | 4 |
| AMP | bf16 |
| Gradient clipping | 0.5 |
| Loss | BCE + IoU |
| Train scales | 0.75, 1.0, 1.25 |
| Augmentation | disabled by default for controlled training |
| Seed | 42 unless otherwise stated |

## MU-Mamba Default Binary Command

```bash
python tools/train_mamba3_seg.py \
  --data-root data/YOUR_DATASET_256_fast \
  --input-mode slices \
  --in-channels 3 \
  --num-classes 1 \
  --image-size 256 \
  --base-dim 32 \
  --depths 2,2,2,2,2 \
  --local-mode dmkdc \
  --mkdc-kernel-sizes 1,3,5 \
  --decoder-skip-bridge guided_axial_seq_grn \
  --epochs 200 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype bf16 \
  --grad-clip 0.5 \
  --no-augment \
  --loss-mode bce_iou \
  --dice-weight 1.0 \
  --bce-weight 1.0 \
  --train-scales 0.75,1.0,1.25 \
  --seed 42 \
  --save-dir outputs/mu_mamba
```

## Evaluation Command

```bash
python tools/eval_mamba3_seg.py \
  --checkpoint outputs/mu_mamba/best.pt \
  --data-root data/YOUR_DATASET_256_fast \
  --split test \
  --pred-dir outputs/mu_mamba/test \
  --image-size 256 \
  --batch-size 1 \
  --num-workers 0 \
  --amp \
  --amp-dtype bf16 \
  --save-probs
```
