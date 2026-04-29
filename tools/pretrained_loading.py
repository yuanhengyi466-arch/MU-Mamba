from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _as_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected checkpoint dict, got {type(checkpoint)!r}")
    for key in (
        "state_dict_ema",
        "state_dict",
        "model_state",
        "model",
        "net",
        "network",
        "module",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "net.", "network.", "backbone."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_partial_pretrained(
    model: nn.Module,
    pretrained_path: str | Path,
    *,
    label: str = "pretrained",
) -> dict[str, int]:
    """Safely load only tensors whose names and shapes exactly match."""

    path = Path(pretrained_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = _as_state_dict(checkpoint)
    target_state = model.state_dict()

    matched: dict[str, torch.Tensor] = {}
    skipped_name = 0
    skipped_shape = 0
    source_tensors = 0
    for raw_key, tensor in source_state.items():
        if not torch.is_tensor(tensor):
            continue
        source_tensors += 1
        key = _strip_prefix(str(raw_key))
        if key not in target_state:
            skipped_name += 1
            continue
        if tuple(tensor.shape) != tuple(target_state[key].shape):
            skipped_shape += 1
            continue
        matched[key] = tensor.detach().cpu()

    missing, unexpected = model.load_state_dict(matched, strict=False)
    summary = {
        "matched": len(matched),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_name": skipped_name,
        "skipped_shape": skipped_shape,
        "source_tensors": source_tensors,
    }
    print(
        f"Loaded {label} weights from {path} "
        f"({summary['matched']} matched tensors, {summary['missing']} missing, "
        f"{summary['unexpected']} unexpected, {summary['skipped_name']} skipped by name, "
        f"{summary['skipped_shape']} skipped by shape)."
    )
    if summary["matched"] == 0:
        print(
            "Warning: no tensors were loaded. This usually means the checkpoint architecture "
            "is incompatible with the current model, or the key names do not match."
        )
    return summary
