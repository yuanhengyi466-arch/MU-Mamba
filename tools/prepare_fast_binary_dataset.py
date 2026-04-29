from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_ssm.data.medical_segmentation import (  # noqa: E402
    _normalize_image,
    _pil_resample,
    _to_2d_image,
    _to_2d_mask,
    collect_paired_cases,
)


def parse_optional_float(value: str | None):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def parse_splits(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed-size fast array cache for a binary segmentation split dataset."
    )
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument(
        "--splits",
        type=parse_splits,
        default=(),
        help="Comma-separated split names. If omitted, existing train/val/test folders are used.",
    )
    parser.add_argument("--window-min", type=parse_optional_float, default=None)
    parser.add_argument("--window-max", type=parse_optional_float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_splits(src_root: Path, requested: tuple[str, ...]) -> tuple[str, ...]:
    if requested:
        splits = requested
    else:
        splits = tuple(split for split in ("train", "val", "test") if (src_root / split / "images").exists())
    if not splits:
        raise FileNotFoundError(f"no split folders found under {src_root}")
    missing = [split for split in splits if not (src_root / split / "images").exists() or not (src_root / split / "masks").exists()]
    if missing:
        raise FileNotFoundError(f"missing images/masks folders for splits: {', '.join(missing)}")
    return splits


def load_image(path: Path, image_size: int, window_min, window_max) -> np.ndarray:
    with Image.open(path) as image:
        image = image.resize((int(image_size), int(image_size)), _pil_resample("BILINEAR"))
        array = np.asarray(image, dtype=np.float32)
    image_array = _normalize_image(_to_2d_image(array), window_min=window_min, window_max=window_max)
    if image_array.ndim == 2:
        image_array = image_array[None, ...]
    return image_array.astype(np.float32)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as mask:
        mask = mask.resize((int(image_size), int(image_size)), _pil_resample("NEAREST"))
        array = np.asarray(mask)
    return _to_2d_mask(array).astype(np.uint8)


def save_split(src_root: Path, out_root: Path, split: str, image_size: int, window_min, window_max) -> dict:
    image_dir = src_root / split / "images"
    mask_dir = src_root / split / "masks"
    pairs = collect_paired_cases(image_dir, mask_dir)
    if not pairs:
        raise ValueError(f"no paired samples found for split={split!r}")

    out_image_dir = out_root / split / "images"
    out_mask_dir = out_root / split / "masks"
    array_dir = out_root / "arrays"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    first_image = load_image(pairs[0].image_path, image_size, window_min, window_max)
    channels = int(first_image.shape[0])
    images = np.empty((len(pairs), channels, image_size, image_size), dtype=np.float32)
    masks = np.empty((len(pairs), 1, image_size, image_size), dtype=np.uint8)
    ids: list[str] = []
    labels: set[int] = set()

    for idx, pair in enumerate(pairs):
        image = first_image if idx == 0 else load_image(pair.image_path, image_size, window_min, window_max)
        mask = load_mask(pair.mask_path, image_size)
        if image.shape != (channels, image_size, image_size):
            raise ValueError(f"{pair.image_path} has shape {image.shape}, expected {(channels, image_size, image_size)}")
        images[idx] = image
        masks[idx, 0] = mask
        ids.append(pair.image_path.stem)
        labels.update(int(value) for value in np.unique(mask).tolist())
        np.save(out_image_dir / f"{pair.image_path.stem}.npy", image, allow_pickle=False)
        np.save(out_mask_dir / f"{pair.mask_path.stem}.npy", mask[None, ...], allow_pickle=False)

    np.save(array_dir / f"{split}_images.npy", images, allow_pickle=False)
    np.save(array_dir / f"{split}_masks.npy", masks, allow_pickle=False)
    (array_dir / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return {
        "split": split,
        "num_samples": len(pairs),
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "labels": sorted(labels),
    }


def main() -> None:
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"source dataset does not exist: {args.src_root}")
    splits = discover_splits(args.src_root, args.splits)
    ensure_clean_dir(args.out_root, overwrite=args.overwrite)

    split_summaries = [
        save_split(args.src_root, args.out_root, split, args.image_size, args.window_min, args.window_max)
        for split in splits
    ]
    summary = {
        "source_root": str(args.src_root.as_posix()),
        "output_root": str(args.out_root.as_posix()),
        "image_size": int(args.image_size),
        "normalization": {
            "type": "same_as_medical_segmentation_loader",
            "window_min": args.window_min,
            "window_max": args.window_max,
        },
        "splits": split_summaries,
    }
    (args.out_root / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        f"# Fast binary dataset {args.image_size}x{args.image_size}",
        "",
        f"Source: `{args.src_root.as_posix()}`",
        "",
        "This folder keeps the same split but stores pre-resized, pre-normalized arrays.",
        "Use it as a drop-in replacement through `--data-root`.",
    ]
    (args.out_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
