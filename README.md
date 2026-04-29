# MU-Mamba

Official PyTorch implementation of **MU-Mamba** for 2D medical image segmentation.

This clean release contains the final **22222 + GRN** configuration used as the main model:

- Five encoder stages with depth configuration `[2, 2, 2, 2, 2]`.
- **DMKDC** local branch with kernel sizes `1, 3, 5`.
- Four-stage U-shaped decoder with four encoder skip connections.
- **DGASF + GRN** skip fusion, implemented as `guided_axial_seq_grn`.


## Structure

```text
mamba_ssm/
  models/vision_mamba3_seg.py        # MU-Mamba model
  data/                              # dataset loaders and fast array datasets
tools/
  train_mamba3_seg.py                # training entry
  eval_mamba3_seg.py                 # evaluation entry
  prepare_fast_binary_dataset.py     # convert image/mask folders to fast arrays
configs/
  binary_segmentation_settings.md    # default experiment settings
```


<img width="5925" height="2920" alt="architecture" src="https://github.com/user-attachments/assets/d2db5849-e4d6-4eb6-a82b-1771449c5133" />


## Installation

Install a CUDA-enabled PyTorch build first, then install the project dependencies.

```bash
conda create -n mu-mamba python=3.10 -y
conda activate mu-mamba

# Example only. Replace this with the correct command for your CUDA version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e . --no-build-isolation
```

The Mamba-3 CUDA path requires a Linux-compatible Triton/CUDA stack. If you are on Windows, using WSL/Linux is recommended.

## Dataset Format

Binary segmentation datasets should follow:

```text
data/DATASET_NAME/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

For faster training, convert a prepared dataset to fixed-size memory-mapped arrays:

```bash
python tools/prepare_fast_binary_dataset.py \
  --data-root data/YOUR_DATASET \
  --out-root data/YOUR_DATASET_256_fast \
  --image-size 256
```

## Train

The training script defaults to the final MU-Mamba configuration. The explicit command is:

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

## Evaluate

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

[acdc_synapse_multiorgan_4x8_qualitative_gh_unet.pdf](https://github.com/user-attachments/files/27211059/acdc_synapse_multiorgan_4x8_qualitative_gh_unet.pdf)
[qualitative_comparison.pdf](https://github.com/user-attachments/files/27211057/qualitative_comparison.pdf)
[polyp_qualitative.pdf](https://github.com/user-attachments/files/27211053/polyp_qualitative.pdf)


## Acknowledgements

This project builds on the upstream Mamba implementation by Tri Dao and Albert Gu. The original Mamba code and CUDA kernels are retained under the Apache-2.0 license.

## License

Apache-2.0.
