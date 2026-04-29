from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba3 import Mamba3


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _make_compatible_groups(channels: int, preferred_groups: int) -> int:
    groups = max(1, min(preferred_groups, channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalResponseNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x.float(), ord=2, dim=(2, 3), keepdim=True)
        normalized = response / (response.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma.to(dtype=x.dtype) * (x * normalized.to(dtype=x.dtype)) + self.beta.to(dtype=x.dtype)


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
                    _make_group_norm(dim),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        return torch.stack(outputs, dim=0).sum(dim=0)


class DynamicMKDCEnhancer(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _make_group_norm(dim),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),
                    _make_group_norm(dim),
                    nn.GELU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        hidden_dim = max(4, dim // 4)
        self.kernel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, len(kernel_sizes), kernel_size=1, bias=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _make_group_norm(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=1)
        weights = torch.softmax(self.kernel_gate(x).flatten(2), dim=1).unsqueeze(2)
        x = (branch_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.project(x)


class _AxialSequenceMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.row_mixer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )
        self.col_mixer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        row = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        row = self.row_mixer(row).reshape(b, h, c, w).permute(0, 2, 1, 3)
        col = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        col = self.col_mixer(col).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return row, col


class GuidedAxialSequenceSkipBridge(nn.Module):
    def __init__(self, skip_channels: int, guide_channels: int) -> None:
        super().__init__()
        self.last_gates: torch.Tensor | None = None
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_channels, skip_channels, kernel_size=1, bias=False),
            _make_group_norm(skip_channels),
            nn.GELU(),
        )
        self.axial = _AxialSequenceMixer(skip_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            _make_group_norm(skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, 2, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels * 4, skip_channels, kernel_size=1, bias=False),
            _make_group_norm(skip_channels),
            nn.GELU(),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(skip_channels),
            nn.GELU(),
        )

    def forward(self, skip: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if guide.shape[-2:] != skip.shape[-2:]:
            guide = F.interpolate(guide, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        guide = self.guide_proj(guide)
        row, col = self.axial(skip + guide)
        gates = torch.sigmoid(self.gate(guide))
        self.last_gates = gates
        row = row * gates[:, 0:1]
        col = col * gates[:, 1:2]
        fused = self.fuse(torch.cat([skip, guide, row, col], dim=1))
        return skip + fused


class GuidedAxialGRNSkipBridge(nn.Module):
    def __init__(self, skip_channels: int, guide_channels: int) -> None:
        super().__init__()
        self.bridge = GuidedAxialSequenceSkipBridge(skip_channels, guide_channels)
        self.grn = GlobalResponseNorm2d(skip_channels)

    @property
    def last_gates(self) -> torch.Tensor | None:
        return self.bridge.last_gates

    def forward(self, skip: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return self.grn(self.bridge(skip, guide))


class TokenMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


def _build_local_enhancer(
    dim: int,
    local_mode: str,
    mkdc_kernel_sizes: Sequence[int],
    d_state: int,
    headdim: int,
    expand: int,
    chunk_size: int,
    dropout: float,
) -> nn.Module:
    return DynamicMKDCEnhancer(dim, kernel_sizes=mkdc_kernel_sizes)


class VisionMamba3Block(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        headdim: int,
        expand: int,
        chunk_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        bidirectional: bool,
        local_mode: str,
        mkdc_kernel_sizes: Sequence[int],
        mamba_branch_mode: str,
    ) -> None:
        super().__init__()
        self.last_local_feature: torch.Tensor | None = None
        self.last_global_feature: torch.Tensor | None = None
        if local_mode != 'dmkdc':
            raise ValueError("the clean MU-Mamba release uses local_mode='dmkdc'")
        if mamba_branch_mode != 'default':
            raise ValueError("the clean MU-Mamba release uses mamba_branch_mode='default'")
        self.bidirectional = bidirectional
        self.mamba_branch_mode = mamba_branch_mode
        self.local = _build_local_enhancer(
            dim=dim,
            local_mode=local_mode,
            mkdc_kernel_sizes=mkdc_kernel_sizes,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            dropout=dropout,
        )
        self.token_norm = nn.LayerNorm(dim)
        self.mamba_fwd = self._build_mamba(
            dim=dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            dropout=dropout,
        )
        if self.bidirectional:
            self.mamba_bwd = self._build_mamba(
                dim=dim,
                d_state=d_state,
                headdim=headdim,
                expand=expand,
                chunk_size=chunk_size,
                dropout=dropout,
            )
            self.merge = nn.Linear(dim * 2, dim)
        else:
            self.merge = nn.Identity()
        self.mlp = TokenMLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def _build_mamba(
        self,
        dim: int,
        d_state: int,
        headdim: int,
        expand: int,
        chunk_size: int,
        dropout: float,
    ) -> Mamba3:
        return Mamba3(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            is_mimo=False,
            is_outproj_norm=False,
            dropout=dropout,
        )

    def _run_scan_branch(
        self,
        tokens: torch.Tensor,
        forward_mamba: Mamba3,
        backward_mamba: Mamba3 | None,
        merge: nn.Module,
    ) -> torch.Tensor:
        normalized = self.token_norm(tokens)
        mixed = forward_mamba(normalized)
        if backward_mamba is not None:
            backward = backward_mamba(normalized.flip(1)).flip(1)
            mixed = merge(torch.cat([mixed, backward], dim=-1))
        else:
            mixed = merge(mixed)
        return mixed

    def forward(self, x: torch.Tensor, return_branches: bool = False) -> torch.Tensor:
        local_out = self.local(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        residual_tokens = tokens
        mixed = self._run_scan_branch(
            tokens,
            self.mamba_fwd,
            self.mamba_bwd if self.bidirectional else None,
            self.merge,
        )
        if return_branches:
            self.last_local_feature = local_out
            self.last_global_feature = mixed.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
        else:
            self.last_local_feature = None
            self.last_global_feature = None
        local_tokens = local_out.flatten(2).transpose(1, 2).contiguous()
        tokens = residual_tokens + self.drop_path(local_tokens)
        tokens = tokens + self.drop_path(mixed)
        tokens = tokens + self.drop_path(self.mlp(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


class VisionMambaStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int,
        headdim: int,
        expand: int,
        chunk_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_paths: Sequence[float],
        bidirectional: bool,
        local_mode: str,
        mkdc_kernel_sizes: Sequence[int],
        mamba_branch_mode: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                VisionMamba3Block(
                    dim=dim,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    chunk_size=chunk_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_paths[idx],
                    bidirectional=bidirectional,
                    local_mode=local_mode,
                    mkdc_kernel_sizes=mkdc_kernel_sizes,
                    mamba_branch_mode=mamba_branch_mode,
                )
                for idx in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_branches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_idx = len(self.blocks) - 1
        for idx, block in enumerate(self.blocks):
            x = block(x, return_branches=return_branches and idx == last_idx)
        if not return_branches:
            return x
        last_block = self.blocks[-1]
        local_feature = last_block.last_local_feature if last_block.last_local_feature is not None else x
        global_feature = last_block.last_global_feature if last_block.last_global_feature is not None else x
        return x, local_feature, global_feature


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        skip_bridge: str = 'none',
    ) -> None:
        super().__init__()
        if skip_bridge != 'guided_axial_seq_grn':
            raise ValueError("the clean MU-Mamba release uses skip_bridge='guided_axial_seq_grn'")
        self.skip_bridge = GuidedAxialGRNSkipBridge(skip_channels, in_channels)
        self.fuse = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.skip_bridge(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x, skip


class VisionMamba3Seg(nn.Module):
    """A trainable 2D medical image segmentation model built on top of Mamba-3."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        dims: Sequence[int] = (32, 64, 128, 256, 512),
        depths: Sequence[int] = (2, 2, 2, 2, 2),
        d_state: int = 64,
        headdim: int = 32,
        expand: int = 2,
        chunk_size: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        bidirectional: bool = True,
        local_mode: str = 'dmkdc',
        stage_local_modes: Sequence[str] | None = None,
        mkdc_kernel_sizes: Sequence[int] = (1, 3, 5),
        decoder_skip_bridge: str = 'guided_axial_seq_grn',
        mamba_branch_mode: str = 'default',
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError('dims and depths must have the same length')
        if len(dims) < 2:
            raise ValueError('at least two stages are required')
        if local_mode != 'dmkdc':
            raise ValueError("the clean MU-Mamba release uses local_mode='dmkdc'")
        if stage_local_modes is None:
            stage_local_modes = tuple(local_mode for _ in dims)
        elif len(stage_local_modes) != len(dims):
            raise ValueError('stage_local_modes must have the same length as dims/depths')
        else:
            stage_local_modes = tuple(stage_local_modes)
        for stage_local_mode in stage_local_modes:
            if stage_local_mode != 'dmkdc':
                raise ValueError("the clean MU-Mamba release uses DMKDC for every stage")
        if decoder_skip_bridge != 'guided_axial_seq_grn':
            raise ValueError("the clean MU-Mamba release uses decoder_skip_bridge='guided_axial_seq_grn'")
        if mamba_branch_mode != 'default':
            raise ValueError("the clean MU-Mamba release uses mamba_branch_mode='default'")
        for dim in dims:
            if (expand * dim) % headdim != 0:
                raise ValueError(
                    f'expand * dim must be divisible by headdim, got dim={dim}, expand={expand}, headdim={headdim}'
                )

        self.deep_supervision = bool(deep_supervision)
        self.stem = ConvBlock(in_channels, dims[0])
        total_blocks = sum(depths)
        if total_blocks == 1:
            drop_path_values = [drop_path_rate]
        else:
            drop_path_values = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        stages = []
        downsamples = []
        cursor = 0
        for idx, (dim, depth) in enumerate(zip(dims, depths)):
            stages.append(
                VisionMambaStage(
                    dim=dim,
                    depth=depth,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    chunk_size=chunk_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_paths=drop_path_values[cursor:cursor + depth],
                    bidirectional=bidirectional,
                    local_mode=stage_local_modes[idx],
                    mkdc_kernel_sizes=mkdc_kernel_sizes,
                    mamba_branch_mode=mamba_branch_mode,
                )
            )
            cursor += depth
            if idx < len(dims) - 1:
                downsamples.append(Downsample(dims[idx], dims[idx + 1]))
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.bottleneck_attention = nn.Identity()

        decoder_blocks = []
        decoder_out_dims = []
        current_dim = dims[-1]
        # Five encoder stages use a standard four-step U-shaped decoder:
        # the bottleneck is decoded through E4/E3/E2/E1 skips, without a
        # self-skip at the deepest stage.
        skip_dims = tuple(reversed(dims[:-1]))
        for skip_dim in skip_dims:
            decoder_blocks.append(
                DecoderBlock(
                    current_dim,
                    skip_dim,
                    skip_dim,
                    skip_bridge=decoder_skip_bridge,
                )
            )
            decoder_out_dims.append(skip_dim)
            current_dim = skip_dim
        self.decoders = nn.ModuleList(decoder_blocks)
        self.head = nn.Sequential(
            ConvBlock(current_dim, current_dim),
            nn.Conv2d(current_dim, num_classes, kernel_size=1),
        )
        if self.deep_supervision and len(decoder_out_dims) > 1:
            self.aux_heads = nn.ModuleList(
                [nn.Conv2d(dim, num_classes, kernel_size=1) for dim in decoder_out_dims[:-1]]
            )
        else:
            self.aux_heads = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        x = self.stem(x)
        skips = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx < len(self.downsamples):
                skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        x = self.bottleneck_attention(x)

        decoder_outputs = []
        for idx, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips))):
            x, _ = decoder(x, skip)
            decoder_outputs.append(x)
        logits = self.head(x)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
        if not self.deep_supervision or len(self.aux_heads) == 0:
            return logits

        aux_logits = []
        for feature_map, head in zip(decoder_outputs[:-1], self.aux_heads):
            aux = head(feature_map)
            if aux.shape[-2:] != output_size:
                aux = F.interpolate(aux, size=output_size, mode='bilinear', align_corners=False)
            aux_logits.append(aux)
        return logits, aux_logits

    def guided_axial_gates(self) -> tuple[torch.Tensor, ...]:
        gates = []
        for decoder in self.decoders:
            skip_bridge = getattr(decoder, 'skip_bridge', None)
            gate = getattr(skip_bridge, 'last_gates', None)
            if gate is not None:
                gates.append(gate)
        return tuple(gates)
