# -*- coding: utf-8 -*-
"""
MRI Reconstruction with Dual-Domain Architecture + Vision Transformer + GAN
============================================================================
Enhanced Architecture (DDC-KSE-ViT-GAN):
- K-space Domain: FFC + UNet + Swin Transformer Blocks
- Image Domain: UNet + ResNet + Swin Transformer Blocks  
- GAN Refinement: PatchGAN Discriminator with Perceptual Loss
- Multi-coil support with sensitivity map estimation


"""
from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"


import time
import datetime
import warnings
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

def gradient_checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
    """Checkpoint wrapper:
    - preserve_rng_state=False to avoid fork_rng/set_rng_state overhead (prevents OOM cascades on Windows)
    - reentrant checkpointing for robustness
    """
    try:
        return _torch_checkpoint(function, *args, use_reentrant=use_reentrant, preserve_rng_state=False)
    except TypeError:
        # Older torch (no preserve_rng_state argument)
        return _torch_checkpoint(function, *args, use_reentrant=use_reentrant)

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler

from collections.abc import Mapping, Sequence
import numbers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# COMPLEX NUMBER & FFT UTILITIES (from original codebase)
# ============================================================
def to_complex_2ch(x2: torch.Tensor) -> torch.Tensor:
    """Convert 2-channel real tensor to complex tensor."""
    return torch.complex(x2[:, 0], x2[:, 1])


def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to 2-channel real tensor."""
    return torch.stack([x.real, x.imag], dim=1)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT with ortho normalization."""
    xc = x.to(torch.complex64)
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.fft2(xc, norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return xc


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT with ortho normalization."""
    xc = x.to(torch.complex64)
    xc = torch.fft.ifftshift(xc, dim=(-2, -1))
    xc = torch.fft.ifft2(xc, norm="ortho")
    xc = torch.fft.fftshift(xc, dim=(-2, -1))
    return xc


def rss_complex(coil_img: torch.Tensor, dim: int = 1, eps: float = 1e-10) -> torch.Tensor:
    """Root Sum of Squares combination for multi-coil images."""
    return torch.sqrt((coil_img.abs() ** 2).sum(dim=dim) + eps)


# ============================================================
# SWIN TRANSFORMER COMPONENTS
# Reference: Swin Transformer (Liu et al., ICCV 2021)
# Adapted for MRI Reconstruction following SwinMR (Huang et al., 2022)
# ============================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C) feature map
        window_size: window size
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H, W: original height and width
        
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA) with relative position bias.
    Reference: Swin Transformer (Liu et al., ICCV 2021)
    
    Args:
        dim: Number of input channels
        window_size: Window size
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to qkv projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size * window_size
            mask: (num_windows, N, N) or None
            
        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
            
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Reference: Swin Transformer (Liu et al., ICCV 2021)
    
    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Window size
        shift_size: Shift size for SW-MSA (0 for W-MSA)
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: If True, add bias to qkv projection
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) where L = H * W
            H, W: spatial dimensions
            
        Returns:
            (B, L, C)
        """
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) doesn't match H*W ({H}*{W})"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def _compute_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Compute attention mask for SW-MSA."""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SwinTransformerStage(nn.Module):
    """
    A stage of Swin Transformer blocks with alternating W-MSA and SW-MSA.
    
    Args:
        dim: Number of input channels
        depth: Number of Swin Transformer blocks
        num_heads: Number of attention heads
        window_size: Local window size
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: If True, add bias to qkv
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate (list or float)
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # Build blocks
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
            
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
            )
            for i in range(depth)
        ])
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) where L = H * W
            H, W: spatial dimensions
            
        Returns:
            (B, L, C)
        """
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using convolutions.
    
    Args:
        in_chans: Number of input channels
        embed_dim: Embedding dimension
        patch_size: Patch size (not used for overlap embedding)
    """
    
    def __init__(self, in_chans: int = 2, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
            
        Returns:
            x: (B, L, embed_dim) where L = H//patch_size * W//patch_size
            H, W: output spatial dimensions
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        x = self.norm(x)
        return x, H, W


class PatchExpand(nn.Module):
    """
    Patch Expanding layer for upsampling in decoder.
    """
    
    def __init__(self, dim: int, dim_scale: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale) if dim_scale == 2 else nn.LayerNorm(dim)
        self.dim_scale = dim_scale
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, L, C)
            H, W: spatial dimensions
            
        Returns:
            x: (B, 4*L, C//2)
            H, W: new spatial dimensions (2*H, 2*W)
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W
        
        x = x.view(B, H, W, C)
        x = x.view(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        
        return x, H * 2, W * 2


# ============================================================
# HYBRID CNN-TRANSFORMER BLOCKS FOR MRI RECONSTRUCTION
# ============================================================

class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridSwinBlock(nn.Module):
    """
    Hybrid CNN + Swin Transformer block for MRI reconstruction.
    Combines local CNN features with global Swin Transformer attention.
    
    Reference: TransUNet (Chen et al., 2021), SwinMR (Huang et al., 2022)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        swin_depth: int = 2,
        use_checkpoint: bool = False,
        swin_downsample: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        
        self.swin_downsample = max(int(swin_downsample), 1)
        # CNN branch for local features
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim, affine=True),
        )
        
        # Swin Transformer branch for global features
        self.swin_stage = SwinTransformerStage(
            dim=dim,
            depth=swin_depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Pad tensor to be divisible by window_size."""
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (0, pad_w, 0, pad_h)
    
    def _unpad(self, x: torch.Tensor, pad: Tuple[int, int, int, int], orig_h: int, orig_w: int) -> torch.Tensor:
        """Remove padding."""
        return x[:, :, :orig_h, :orig_w]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # CNN branch
        conv_out = self.conv_branch(x)
        
        # Swin branch (need to handle padding for window attention)
        x_swin_in = x
        Hs, Ws = H, W
        if self.swin_downsample > 1:
            x_swin_in = F.avg_pool2d(x, kernel_size=self.swin_downsample, stride=self.swin_downsample)
            Hs, Ws = int(x_swin_in.shape[-2]), int(x_swin_in.shape[-1])

        x_padded, pad_info = self._pad_to_window(x_swin_in)
        _, _, Hp, Wp = x_padded.shape

        # Reshape for Swin: (B, C, H, W) -> (B, H*W, C)
        x_swin = x_padded.flatten(2).transpose(1, 2)
        # Apply Swin Transformer
        if self.use_checkpoint and self.training:
            x_swin = gradient_checkpoint(self.swin_stage, x_swin, Hp, Wp, use_reentrant=True)
        else:
            x_swin = self.swin_stage(x_swin, Hp, Wp)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_swin = x_swin.transpose(1, 2).view(B, C, Hp, Wp)
        x_swin = self._unpad(x_swin, pad_info, Hs, Ws)
        if self.swin_downsample > 1:
            x_swin = F.interpolate(x_swin, size=(H, W), mode='bilinear', align_corners=False)
        # Fusion
        fused = self.fusion(torch.cat([conv_out, x_swin], dim=1))
        
        return self.act(shortcut + fused)


# ============================================================
# SPECTRAL TRANSFORM (FFC) - From Original Codebase
# ============================================================

class SpectralTransform(nn.Module):
    """Spectral transform using FFT for global context in FFC."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels * 2, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE:
        # cuFFT has restrictions for half precision FFTs (fp16/bf16). In particular, for rfft2/irfft2
        # it may require power-of-two spatial sizes when computing in half precision.
        # fastMRI slices often have shapes like 640x320, which will crash in fp16/bf16.
        # Therefore we ALWAYS run FFT/iFFT in fp32 for robustness, while keeping the spectral
        # convolution path optionally in fp16 to reduce memory.
        orig_dtype = x.dtype

        # Always compute FFT in fp32
        x_fft32 = x.float()
        batch = x_fft32.shape[0]

        # FFT (fp32 -> complex64)
        ffted = torch.fft.rfft2(x_fft32, norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)  # float32 [B,C,H,Wf,2]

        # To conv layout: [B, 2C, H, Wf]
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, ffted.shape[-2], ffted.shape[-1])

        # Spectral convs can be done in fp16 for memory even if orig dtype is bf16
        conv_dtype = torch.float16 if orig_dtype in (torch.float16, torch.bfloat16) else torch.float32
        ffted = ffted.to(conv_dtype)

        ffted = self.conv1(ffted)
        ffted = self.bn(ffted)
        ffted = self.relu(ffted)
        ffted = self.conv2(ffted)

        # Back to fp32 for iFFT
        ffted = ffted.to(torch.float32)
        ffted = ffted.view(batch, -1, 2, ffted.shape[-2], ffted.shape[-1])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=x_fft32.shape[-2:], norm="ortho")  # float32

        if orig_dtype in (torch.float16, torch.bfloat16):
            output = output.to(orig_dtype)
        return output




class FFCBlock(nn.Module):
    """Fast Fourier Convolution block with local and global branches."""
    
    def __init__(self, in_channels: int, out_channels: int, ratio_global: float = 0.5, k: int = 3):
        super().__init__()
        in_local = max(int(in_channels * (1 - ratio_global)), 1)
        in_global = max(in_channels - in_local, 1)
        out_local = max(int(out_channels * (1 - ratio_global)), 1)
        out_global = max(out_channels - out_local, 1)
        self.in_local = in_local
        self.out_local = out_local
        pad = k // 2

        self.l2l = nn.Sequential(
            nn.Conv2d(in_local, out_local, k, padding=pad),
            nn.InstanceNorm2d(out_local, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.l2g = nn.Sequential(
            nn.Conv2d(in_local, out_global, k, padding=pad),
            nn.InstanceNorm2d(out_global, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.g2l = nn.Sequential(
            nn.Conv2d(in_global, out_local, k, padding=pad),
            nn.InstanceNorm2d(out_local, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.g2g = SpectralTransform(in_global, out_global)
        self.g2g_bn = nn.InstanceNorm2d(out_global, affine=True)
        self.g2g_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl, xg = x[:, :self.in_local], x[:, self.in_local:]
        yl = self.l2l(xl) + self.g2l(xg)
        yg = self.l2g(xl) + self.g2g_act(self.g2g_bn(self.g2g(xg)))
        return torch.cat([yl, yg], dim=1)


class FFCResBlock(nn.Module):
    """Residual block using FFC."""
    
    def __init__(self, ch: int, ratio: float = 0.5, scale: float = 0.3):
        super().__init__()
        self.a = FFCBlock(ch, ch, ratio)
        self.b = FFCBlock(ch, ch, ratio)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.b(self.a(x))


# ============================================================
# K-SPACE DOMAIN NETWORK WITH SWIN TRANSFORMER
# ============================================================

class KSpaceFFCUNetSwin(nn.Module):
    """
    K-space domain network combining FFC + UNet + Swin Transformer.
    
    Architecture follows SwinMR principles:
    - Encoder: Conv + FFC + Swin Transformer for multi-scale features
    - Bottleneck: Swin Transformer for global context
    - Decoder: Conv + Swin Transformer with skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_ch: int = 48,
        num_ffc_blocks: int = 2,
        ffc_ratio: float = 0.5,
        swin_depths: Tuple[int, ...] = (2, 2, 2),
        swin_heads: Tuple[int, ...] = (3, 6, 12),
        window_size: int = 8,
        coil_chunk: int = 4,
        use_checkpoint: bool = True,
        residual_scale: float = 0.3,
    ):
        super().__init__()
        self.coil_chunk = max(int(coil_chunk), 1)
        self.use_checkpoint = bool(use_checkpoint)
        self.residual_scale = residual_scale
        self.window_size = window_size
        
        # Initial convolution
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Encoder Stage 1: FFC + Swin
        self.enc1_ffc = nn.Sequential(*[FFCResBlock(base_ch, ffc_ratio, scale=0.3) for _ in range(num_ffc_blocks)])
        self.enc1_swin = HybridSwinBlock(base_ch, num_heads=swin_heads[0], window_size=window_size, 
                                          swin_depth=swin_depths[0], use_checkpoint=use_checkpoint, swin_downsample=4)
        self.down1 = nn.Sequential(nn.AvgPool2d(2), ConvBlock(base_ch, base_ch * 2))
        
        # Encoder Stage 2: FFC + Swin
        self.enc2_ffc = nn.Sequential(*[FFCResBlock(base_ch * 2, ffc_ratio, scale=0.3) for _ in range(num_ffc_blocks)])
        self.enc2_swin = HybridSwinBlock(base_ch * 2, num_heads=swin_heads[1], window_size=window_size,
                                          swin_depth=swin_depths[1], use_checkpoint=use_checkpoint)
        self.down2 = nn.Sequential(nn.AvgPool2d(2), ConvBlock(base_ch * 2, base_ch * 4))
        
        # Bottleneck: Pure Swin Transformer for global context
        self.bottleneck_swin = HybridSwinBlock(base_ch * 4, num_heads=swin_heads[2], window_size=window_size,
                                                swin_depth=swin_depths[2], use_checkpoint=use_checkpoint)
        
        # Decoder
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2_conv = ConvBlock(base_ch * 4, base_ch * 2)  # After concat with skip
        self.dec2_swin = HybridSwinBlock(base_ch * 2, num_heads=swin_heads[1], window_size=window_size,
                                          swin_depth=swin_depths[1], use_checkpoint=use_checkpoint)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1_conv = ConvBlock(base_ch * 2, base_ch)  # After concat with skip
        self.dec1_swin = HybridSwinBlock(base_ch, num_heads=swin_heads[0], window_size=window_size,
                                          swin_depth=swin_depths[0], use_checkpoint=use_checkpoint, swin_downsample=4)
        
        # Output convolution
        self.outc = nn.Conv2d(base_ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.outc.bias)
        nn.init.normal_(self.outc.weight, std=0.02)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single chunk."""
        # Initial
        x0 = self.inc(x)
        
        # Encoder
        e1 = self.enc1_ffc(x0)
        e1 = self.enc1_swin(e1)
        d1 = self.down1(e1)
        
        e2 = self.enc2_ffc(d1)
        e2 = self.enc2_swin(e2)
        d2 = self.down2(e2)
        
        # Bottleneck
        b = self.bottleneck_swin(d2)
        
        # Decoder with skip connections
        u2 = self.up2(b)
        # Handle size mismatch
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2_conv(u2)
        u2 = self.dec2_swin(u2)
        
        u1 = self.up1(u2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1_conv(u1)
        u1 = self.dec1_swin(u1)
        
        out = self.outc(u1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with coil chunking for memory efficiency."""
        if x.dim() != 4 or x.size(1) != 2:
            raise ValueError(f"Expected [N, 2, H, W], got {tuple(x.shape)}")
        x_work = x
        outs = []

        for s in range(0, x_work.shape[0], self.coil_chunk):
            xs = x_work[s:s + self.coil_chunk]
            if self.use_checkpoint and self.training:
                ys = gradient_checkpoint(self._forward_single, xs, use_reentrant=True)
            else:
                ys = self._forward_single(xs)
            outs.append(xs + self.residual_scale * ys)

        return torch.cat(outs, dim=0).to(x.dtype)


# ============================================================
# IMAGE DOMAIN NETWORK WITH SWIN TRANSFORMER
# ============================================================

class ImageUNetResNetSwin(nn.Module):
    """
    Image domain network combining UNet + ResNet + Swin Transformer.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_ch: int = 32,
        num_resblocks: int = 2,
        depth: int = 3,
        swin_depths: Tuple[int, ...] = (2, 2, 2),
        swin_heads: Tuple[int, ...] = (2, 4, 8),
        window_size: int = 8,
        residual_scale: float = 0.3,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.residual_scale = residual_scale
        self.use_checkpoint = use_checkpoint
        
        # Initial conv
        self.inc = ConvBlock(in_channels, base_ch)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.encoder_swins = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        
        for i in range(depth):
            self.encoders.append(
                nn.Sequential(*[ResNetBlock(ch) for _ in range(num_resblocks)])
            )
            heads = swin_heads[min(i, len(swin_heads) - 1)]
            swin_d = swin_depths[min(i, len(swin_depths) - 1)]
            self.encoder_swins.append(
                HybridSwinBlock(ch, num_heads=heads, window_size=window_size, 
                               swin_depth=swin_d, use_checkpoint=use_checkpoint, swin_downsample=(4 if i == 0 else 1))
            )
            next_ch = min(ch * 2, base_ch * 8)
            self.downs.append(nn.Sequential(nn.AvgPool2d(2), ConvBlock(ch, next_ch)))
            ch = next_ch
        
        # Bottleneck with Swin Transformer
        self.bottleneck = nn.Sequential(*[ResNetBlock(ch) for _ in range(num_resblocks * 2)])
        self.bottleneck_swin = HybridSwinBlock(
            ch, num_heads=swin_heads[-1], window_size=window_size,
            swin_depth=swin_depths[-1], use_checkpoint=use_checkpoint
        )
        
        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_swins = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            prev_ch = ch
            ch = base_ch * (2 ** i) if i > 0 else base_ch
            ch = min(ch, base_ch * 8)
            skip_ch = base_ch * (2 ** i) if i > 0 else base_ch
            skip_ch = min(skip_ch, base_ch * 8)
            
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev_ch, ch, 3, padding=1),
                nn.InstanceNorm2d(ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.decoders.append(
                nn.Sequential(
                    ConvBlock(ch + skip_ch, ch),
                    *[ResNetBlock(ch) for _ in range(num_resblocks)]
                )
            )
            heads = swin_heads[min(depth - 1 - i, len(swin_heads) - 1)]
            swin_d = swin_depths[min(depth - 1 - i, len(swin_depths) - 1)]
            self.decoder_swins.append(
                HybridSwinBlock(ch, num_heads=heads, window_size=window_size,
                               swin_depth=swin_d, use_checkpoint=use_checkpoint, swin_downsample=(4 if i == 0 else 1))
            )
        
        # Output conv
        self.outc = nn.Conv2d(base_ch, in_channels, 1)
        nn.init.zeros_(self.outc.bias)
        nn.init.normal_(self.outc.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.inc(x)
        
        # Encoder
        skips = []
        for enc, enc_swin, down in zip(self.encoders, self.encoder_swins, self.downs):
            x = enc(x)
            x = enc_swin(x)
            skips.append(x)
            x = down(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.bottleneck_swin(x)
        
        # Decoder
        for up, dec, dec_swin, skip in zip(self.ups, self.decoders, self.decoder_swins, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            x = dec_swin(x)
        
        out = self.outc(x)
        return inp + self.residual_scale * out


class ResNetBlock(nn.Module):
    """ResNet-style residual block."""
    
    def __init__(self, channels: int, k: int = 3, scale: float = 1.0):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=p),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, k, padding=p),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.scale * self.net(x))


# ============================================================
# GAN DISCRIMINATOR (PatchGAN)
# Reference: DAGAN (Yang et al., 2018), pix2pix (Isola et al., 2017)
# ============================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for MRI reconstruction.
    
    Outputs a grid of real/fake predictions, providing more spatially
    localized feedback than a single global discriminator output.
    
    Reference: pix2pix (Isola et al., CVPR 2017)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        norm_layer = nn.InstanceNorm2d
        
        def get_conv(in_ch, out_ch, kernel_size=4, stride=2, padding=1, use_sn=use_spectral_norm):
            conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            if use_sn:
                conv = nn.utils.spectral_norm(conv)
            return conv
        
        sequence = [
            get_conv(in_channels, base_ch),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                get_conv(base_ch * nf_mult_prev, base_ch * nf_mult),
                norm_layer(base_ch * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            get_conv(base_ch * nf_mult_prev, base_ch * nf_mult, stride=1),
            norm_layer(base_ch * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Output layer
        sequence += [get_conv(base_ch * nf_mult, 1, stride=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image (real or reconstructed)
            
        Returns:
            (B, 1, H', W') patch-wise real/fake predictions
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN Discriminator.
    
    Uses discriminators at multiple scales for better gradient flow
    and capturing both fine and coarse structures.
    
    Reference: pix2pixHD (Wang et al., CVPR 2018)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 64,
        n_layers: int = 3,
        num_discriminators: int = 2,
    ):
        super().__init__()
        self.num_discriminators = num_discriminators
        
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            self.discriminators.append(
                PatchDiscriminator(in_channels, base_ch, n_layers)
            )
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input image
            
        Returns:
            List of discriminator outputs at different scales
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        return outputs


# ============================================================
# PERCEPTUAL LOSS (VGG-based)
# Reference: Johnson et al., ECCV 2016
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for image quality.
    
    Uses features from pretrained VGG19 to compare perceptual similarity
    between reconstructed and ground truth images.
    """
    
    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 15, 22],
        use_input_norm: bool = True,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except ImportError:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        if weights is None:
            weights = [1.0 / len(feature_layers)] * len(feature_layers)
        self.weights = weights
        
        # Extract VGG layers
        self.vgg_layers = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            self.vgg_layers.append(nn.Sequential(*list(vgg.children())[prev_layer:layer_idx + 1]))
            prev_layer = layer_idx + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        if self.use_input_norm:
            mean = self.mean.to(device=x.device, dtype=x.dtype)
            std = self.std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        return x
    
    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale to RGB."""
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predicted image
            target: (B, C, H, W) target image
            
        Returns:
            Perceptual loss (scalar)
        """
        pred = self._to_rgb(pred)
        target = self._to_rgb(target)
        
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        loss = 0.0
        pred_feat = pred
        target_feat = target
        
        for i, vgg_layer in enumerate(self.vgg_layers):
            pred_feat = vgg_layer(pred_feat)
            with torch.no_grad():
                target_feat = vgg_layer(target_feat)
            loss += self.weights[i] * F.l1_loss(pred_feat, target_feat)
        
        return loss


# ============================================================
# GAN REFINEMENT MODULE
# ============================================================

class GANRefinementModule(nn.Module):
    """
    GAN-based refinement module for final image enhancement.
    
    Takes the output of the main reconstruction network and refines it
    using adversarial training for improved perceptual quality.
    
    Reference: RefineGAN (Quan et al., 2018), DAGAN (Yang et al., 2018)
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_ch: int = 64,
        num_residual_blocks: int = 6,
        use_swin: bool = True,
        swin_heads: int = 4,
        window_size: int = 8,
    ):
        super().__init__()
        
        self.use_swin = use_swin
        
        # Initial conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 7, padding=3),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks with optional Swin Transformer
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResNetBlock(base_ch * 4))
        
        if use_swin:
            self.swin_block = HybridSwinBlock(
                base_ch * 4, num_heads=swin_heads, window_size=window_size, swin_depth=2
            )
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, 7, padding=3),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        
        self.residual_scale = 0.3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image from main reconstruction
            
        Returns:
            (B, C, H, W) refined image
        """
        shortcut = x
        
        # Encoder
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Residual processing
        for block in self.residual_blocks:
            x = block(x)
        
        if self.use_swin:
            x = self.swin_block(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv_out(x)
        
        # Residual connection
        return shortcut + self.residual_scale * x


# ============================================================
# DATA CONSISTENCY MODULE (from original codebase)
# ============================================================

def _ensure_mask4(mask: torch.Tensor) -> torch.Tensor:
    """Ensure sampling mask shape is [B,1,H,W]."""
    if mask is None:
        return mask
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}")

    if mask.dim() == 5 and mask.size(2) == 1:
        mask = mask.squeeze(2)
    elif mask.dim() == 5 and mask.size(1) == 1 and mask.size(2) != 1:
        while mask.dim() > 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)

    if mask.dim() == 4:
        return mask
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def estimate_sens_from_kspace(k_meas: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Estimate sensitivity maps from k-space."""
    if k_meas.dim() == 3:
        k_meas = k_meas.unsqueeze(0)
    elif k_meas.dim() == 5 and k_meas.size(1) == 1:
        k_meas = k_meas.squeeze(1)

    k_meas = k_meas.to(torch.complex64)
    coil_img = ifft2c(k_meas)
    rss = torch.sqrt((coil_img.abs() ** 2).sum(dim=1, keepdim=True) + eps).clamp_min(eps)
    sens = coil_img / rss
    norm = torch.sqrt((sens.abs() ** 2).sum(dim=1, keepdim=True) + eps).clamp_min(eps)
    sens = sens / norm
    return sens.to(torch.complex64)


def AH(k: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """Adjoint operator A^H: k-space -> coil-combined image."""
    if k.dim() == 3:
        k = k.unsqueeze(0)
    if sens.dim() == 3:
        sens = sens.unsqueeze(0)
    if k.dim() == 5:
        k = k.squeeze(1)
    if sens.dim() == 5:
        sens = sens.squeeze(1)
    
    x_coils = ifft2c(k)
    x_combined = (torch.conj(sens) * x_coils).sum(dim=1)
    return x_combined


class DataConsistencyMC(nn.Module):
    """Multi-coil data consistency with hard replacement."""
    
    def __init__(self, learnable: bool = False):
        super().__init__()
        self.learnable = learnable

    def forward(self, x_img: torch.Tensor, k_meas: torch.Tensor, mask: torch.Tensor, sens: torch.Tensor):
        if k_meas.dim() == 5:
            k_meas = k_meas.squeeze(1)
        if sens.dim() == 5:
            sens = sens.squeeze(1)
        if sens.dim() == 3:
            sens = sens.unsqueeze(0)

        if torch.is_tensor(x_img) and x_img.dim() == 4:
            x_img = (torch.conj(sens) * x_img).sum(dim=1)
        if x_img.dim() == 2:
            x_img = x_img.unsqueeze(0)

        x_img = x_img.to(torch.complex64)
        k_meas = k_meas.to(torch.complex64)
        sens = sens.to(torch.complex64)

        k_pred = fft2c(sens * x_img[:, None])
        mask4 = _ensure_mask4(mask)
        m = mask4[:, 0].to(dtype=k_pred.dtype)[:, None]

        k_dc = m * k_meas + (1.0 - m) * k_pred
        x_dc = AH(k_dc, sens)

        return x_dc


# ============================================================
# DUAL-DOMAIN CASCADE WITH SWIN TRANSFORMER
# ============================================================

class DualDomainCascadeSwin(nn.Module):
    """
    Single cascade of dual-domain reconstruction with Swin Transformer.
    """
    
    def __init__(
        self,
        k_base_ch: int = 32,
        k_ffc_blocks: int = 2,
        k_ffc_ratio: float = 0.5,
        k_swin_depths: Tuple[int, ...] = (1, 1, 1),
        k_swin_heads: Tuple[int, ...] = (2, 4, 8),
        k_window_size: int = 8,
        k_coil_chunk: int = 1,
        k_use_ckpt: bool = True,
        k_residual_scale: float = 0.3,
        img_base_ch: int = 24,
        img_resblocks: int = 1,
        img_depth: int = 2,
        img_swin_depths: Tuple[int, ...] = (1, 1, 1),
        img_swin_heads: Tuple[int, ...] = (2, 4, 8),
        img_window_size: int = 8,
        img_residual_scale: float = 0.3,
        img_use_ckpt: bool = True,
    ):
        super().__init__()
        
        # K-space domain: FFC + UNet + Swin
        self.k_refine = KSpaceFFCUNetSwin(
            in_channels=2,
            base_ch=k_base_ch,
            num_ffc_blocks=k_ffc_blocks,
            ffc_ratio=k_ffc_ratio,
            swin_depths=k_swin_depths,
            swin_heads=k_swin_heads,
            window_size=k_window_size,
            coil_chunk=k_coil_chunk,
            use_checkpoint=k_use_ckpt,
            residual_scale=k_residual_scale,
        )
        
        self.dc_k = DataConsistencyMC()
        
        # Image domain: UNet + ResNet + Swin
        self.img_refine = ImageUNetResNetSwin(
            in_channels=2,
            base_ch=img_base_ch,
            num_resblocks=img_resblocks,
            depth=img_depth,
            swin_depths=img_swin_depths,
            swin_heads=img_swin_heads,
            window_size=img_window_size,
            residual_scale=img_residual_scale,
            use_checkpoint=img_use_ckpt,
        )
        
        self.dc_img = DataConsistencyMC()

    def forward(
        self,
        x_img: torch.Tensor,
        k_meas: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
    ) -> torch.Tensor:
        # ---- shape sanitation ----
        if k_meas.dim() == 5:
            k_meas = k_meas.squeeze(1)
        if sens.dim() == 5:
            sens = sens.squeeze(1)
        if sens.dim() == 3:
            sens = sens.unsqueeze(0)

        # If x_img is coil-stacked, combine with sensitivities
        if x_img.dim() == 4:
            with torch.cuda.amp.autocast(enabled=False):
                x_img = (torch.conj(sens) * x_img).sum(dim=1)
        if x_img.dim() == 2:
            x_img = x_img.unsqueeze(0)

        # ---- K-space refinement (keep FFT/DC in fp32, allow k_refine in AMP) ----
        with torch.cuda.amp.autocast(enabled=False):
            k_pred = fft2c(sens * x_img[:, None])                     # [B,Nc,H,W] complex64
            k2 = torch.stack([k_pred.real, k_pred.imag], dim=2)       # [B,Nc,2,H,W] fp32
            B, Nc, C2, H, W = k2.shape
            k2_in = k2.reshape(B * Nc, C2, H, W).contiguous().float() # [B*Nc,2,H,W]

        # Run k_refine under autocast if enabled upstream
        k2_ref = self.k_refine(k2_in)                                 # [B*Nc,2,H,W]

        with torch.cuda.amp.autocast(enabled=False):
            k2_ref = k2_ref.float().reshape(B, Nc, C2, H, W)
            k_ref = torch.complex(k2_ref[:, :, 0], k2_ref[:, :, 1])    # [B,Nc,H,W] complex64
            x_from_k = AH(k_ref, sens)                                 # [B,H,W] complex64
            x1 = self.dc_k(x_from_k, k_meas, mask, sens)

        # ---- Image refinement (AMP-friendly) ----
        x2 = complex_to_2ch(x1)                                        # [B,2,H,W]
        x2 = self.img_refine(x2)
        x_img2 = to_complex_2ch(x2)

        with torch.cuda.amp.autocast(enabled=False):
            x_out = self.dc_img(x_img2, k_meas, mask, sens)

        return x_out



# ============================================================
# COMPLETE MODEL: DDC-KSE-ViT-GAN
# ============================================================

class DualDomainMRIReconstructionViTGAN(nn.Module):
    """
    Complete Dual-Domain MRI Reconstruction with Vision Transformer and GAN.
    
    Architecture (DDC-KSE-ViT-GAN):
    1. Cascaded dual-domain reconstruction with Swin Transformer
    2. GAN refinement module for perceptual quality enhancement
    
    This architecture follows academic best practices from:
    - SwinMR (Huang et al., 2022) for Swin Transformer integration
    - E2E-VarNet (Sriram et al., 2020) for cascaded refinement
    - DAGAN (Yang et al., 2018) for GAN-based enhancement
    """
    
    def __init__(
        self,
        num_cascades: int = 2,
        # K-space network config
        k_base_ch: int = 48,
        k_ffc_blocks: int = 2,
        k_ffc_ratio: float = 0.5,
        k_swin_depths: Tuple[int, ...] = (2, 2, 2),
        k_swin_heads: Tuple[int, ...] = (3, 6, 12),
        k_window_size: int = 8,
        k_coil_chunk: int = 4,
        k_use_ckpt: bool = True,
        k_residual_scale: float = 0.3,
        # Image network config
        img_base_ch: int = 32,
        img_resblocks: int = 2,
        img_depth: int = 3,
        img_swin_depths: Tuple[int, ...] = (2, 2, 2),
        img_swin_heads: Tuple[int, ...] = (2, 4, 8),
        img_window_size: int = 8,
        img_residual_scale: float = 0.3,
        img_use_ckpt: bool = True,
        # GAN config
        use_gan_refinement: bool = True,
        gan_base_ch: int = 64,
        gan_residual_blocks: int = 6,
        gan_use_swin: bool = True,
    ):
        super().__init__()
        
        self.num_cascades = num_cascades
        self.use_gan_refinement = use_gan_refinement
        
        # Cascaded dual-domain reconstruction
        self.cascades = nn.ModuleList([
            DualDomainCascadeSwin(
                k_base_ch=k_base_ch,
                k_ffc_blocks=k_ffc_blocks,
                k_ffc_ratio=k_ffc_ratio,
                k_swin_depths=k_swin_depths,
                k_swin_heads=k_swin_heads,
                k_window_size=k_window_size,
                k_coil_chunk=k_coil_chunk,
                k_use_ckpt=k_use_ckpt,
                k_residual_scale=k_residual_scale,
                img_base_ch=img_base_ch,
                img_resblocks=img_resblocks,
                img_depth=img_depth,
                img_swin_depths=img_swin_depths,
                img_swin_heads=img_swin_heads,
                img_window_size=img_window_size,
                img_residual_scale=img_residual_scale,
                img_use_ckpt=img_use_ckpt,
            )
            for _ in range(num_cascades)
        ])
        
        # GAN refinement module
        if use_gan_refinement:
            self.gan_refine = GANRefinementModule(
                in_channels=2,
                base_ch=gan_base_ch,
                num_residual_blocks=gan_residual_blocks,
                use_swin=gan_use_swin,
            )

    def forward(
        self,
        k_meas: torch.Tensor,
        mask: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through complete model.
        
        Args:
            k_meas: [B, Nc, H, W] complex measured k-space
            mask: [B, 1, H, W] sampling mask
            return_intermediate: whether to return intermediate cascade outputs
            
        Returns:
            x: [B, H, W] complex final reconstruction
            intermediates: (optional) list of intermediate outputs
        """
        if k_meas.dim() == 3:
            k_meas = k_meas.unsqueeze(0)
        if k_meas.dim() == 5:
            k_meas = k_meas.squeeze(1)
        
        if mask.dim() == 5 and mask.shape[2] == 1:
            mask = mask.squeeze(2)

        # Pad to window size multiple
        k_meas, _pad = self._pad_to_window(k_meas)
        if _pad != (0, 0, 0, 0):
            pl, pr, pt, pb = _pad
            mask = F.pad(mask, (pl, pr, pt, pb))

        with torch.cuda.amp.autocast(enabled=False):
            sens = estimate_sens_from_kspace(k_meas.to(torch.complex64))
            x = AH(k_meas.to(torch.complex64), sens)
        
        intermediates = []
        for cascade in self.cascades:
            x = cascade(x, k_meas, mask, sens)
            if return_intermediate:
                intermediates.append(x.clone())
        
        # GAN refinement
        if self.use_gan_refinement:
            x_2ch = complex_to_2ch(x)
            x_refined = self.gan_refine(x_2ch)
            x = to_complex_2ch(x_refined)
        
        # Unpad
        if _pad != (0, 0, 0, 0):
            x = self._unpad(x, _pad)
        
        if return_intermediate:
            return x, intermediates
        return x
    
    def _pad_to_window(self, x: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h = (mult - (h % mult)) % mult
        pad_w = (mult - (w % mult)) % mult
        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        return F.pad(x, (pl, pr, pt, pb)), (pl, pr, pt, pb)
    
    def _unpad(self, x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
        pl, pr, pt, pb = pad
        if (pl, pr, pt, pb) == (0, 0, 0, 0):
            return x
        h, w = x.shape[-2], x.shape[-1]
        return x[..., pt:h - pb, pl:w - pr]


# ============================================================
# GAN TRAINING UTILITIES
# ============================================================

class GANLoss(nn.Module):
    """
    GAN loss with support for different GAN types.
    
    Supports:
    - vanilla: standard GAN loss (BCE)
    - lsgan: least squares GAN loss (MSE)
    - wgan: Wasserstein GAN loss
    - hinge: hinge loss
    """
    
    def __init__(self, gan_mode: str = 'lsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgan', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        # ✅ kritik: aynı device + AMP için aynı dtype
        target_tensor = target_tensor.to(device=prediction.device, dtype=prediction.dtype)
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1.0 - prediction).mean()
            else:
                loss = F.relu(1.0 + prediction).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for GAN-based MRI reconstruction.
    
    Loss = λ_l1 * L1 + λ_ssim * SSIM + λ_perceptual * Perceptual + λ_adv * Adversarial
    
    Reference: DAGAN (Yang et al., 2018), RefineGAN (Quan et al., 2018)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.01,
        gan_mode: str = 'lsgan',
        use_perceptual: bool = True,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.use_perceptual = use_perceptual
        
        # SSIM loss
        self.ssim_loss = SSIMLoss()
        
        # Perceptual loss
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()
        
        # Adversarial loss
        self.gan_loss = GANLoss(gan_mode)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_pred_fake: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: predicted magnitude image
            target: target magnitude image
            disc_pred_fake: discriminator prediction on fake image (optional)
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # L1 loss
        losses['l1'] = F.l1_loss(pred, target)
        
        # SSIM loss
        losses['ssim'] = self.ssim_loss(pred, target)
        
        # Perceptual loss
        if self.use_perceptual and hasattr(self, 'perceptual_loss'):
            losses['perceptual'] = self.perceptual_loss(pred, target)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        
        # Adversarial loss
        if disc_pred_fake is not None:
            if isinstance(disc_pred_fake, list):
                adv_loss = 0.0
                for pred_i in disc_pred_fake:
                    adv_loss += self.gan_loss(pred_i, True)
                losses['adversarial'] = adv_loss / len(disc_pred_fake)
            else:
                losses['adversarial'] = self.gan_loss(disc_pred_fake, True)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=pred.device)
        
        # Total loss
        losses['total'] = (
            self.l1_weight * losses['l1'] +
            self.ssim_weight * losses['ssim'] +
            self.perceptual_weight * losses['perceptual'] +
            self.adversarial_weight * losses['adversarial']
        )
        
        return losses


class SSIMLoss(nn.Module):
    """SSIM loss for training."""
    
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.win_size = win_size
        self.k1 = k1
        self.k2 = k2
        
        coords = torch.arange(win_size, dtype=torch.float32) - (win_size // 2)
        g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
        g = g / g.sum()
        window = (g[:, None] * g[None, :])[None, None]
        self.register_buffer("window", window)
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred[:, None]
        if gt.dim() == 3:
            gt = gt[:, None]
            
        data_range = (gt.amax(dim=(-2, -1), keepdim=True) - gt.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-10)
        
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        
        pad = self.win_size // 2
        window = self.window.to(pred.device, pred.dtype)
        
        mu_x = F.conv2d(gt, window, padding=pad)
        mu_y = F.conv2d(pred, window, padding=pad)
        
        mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        
        sigma_x_sq = F.conv2d(gt ** 2, window, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(pred ** 2, window, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(gt * pred, window, padding=pad) - mu_xy
        
        sigma_x_sq = sigma_x_sq.clamp_min(0.0)
        sigma_y_sq = sigma_y_sq.clamp_min(0.0)
        
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12)
        
        return 1.0 - ssim_map.mean()


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainConfigViTGAN:
    """Training configuration for ViT + GAN model."""
    train_path: str
    val_path: str
    test_path: str = ""
    out_dir: str = "./outputs_vit_gan"
    
    # Data
    batch_size: int = 1
    num_workers: int = 4
    accelerations: Tuple[int, ...] = (4,)
    cf_map: Tuple[Tuple[int, float], ...] = ((4, 0.08),)
    eval_crop_hw: Tuple[int, int] = (320, 320)
    # Model - Reconstruction
    num_cascades: int = 2
    k_base_ch: int = 32
    k_ffc_blocks: int = 2
    k_swin_depths: Tuple[int, ...] = (1, 1, 1)
    k_swin_heads: Tuple[int, ...] = (2, 4, 8)
    k_window_size: int = 8
    k_coil_chunk: int = 1
    img_base_ch: int = 24
    img_resblocks: int = 1
    img_depth: int = 2
    img_swin_depths: Tuple[int, ...] = (1, 1, 1)
    img_swin_heads: Tuple[int, ...] = (2, 4, 8)
    img_window_size: int = 8
    
    # Model - GAN
    use_gan_refinement: bool = True
    gan_base_ch: int = 64
    gan_residual_blocks: int = 6
    gan_use_swin: bool = True
    disc_base_ch: int = 64
    disc_n_layers: int = 3
    num_discriminators: int = 2
    
    # Training
    epochs: int = 50
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    grad_clip: float = 1.0
    

    empty_cache_every: int = 50  # torch.cuda.empty_cache() every N steps (0=off)
    skip_oom: bool = True        # skip batch on CUDA OOM
    # Loss weights
    l1_weight: float = 1.0
    ssim_weight: float = 0.1
    perceptual_weight: float = 0.1
    adversarial_weight: float = 0.01
    
    # GAN training
    gan_start_epoch: int = 5  # Start GAN training after N epochs of pure reconstruction
    d_steps_per_g: int = 1
    
    # AMP
    use_amp: bool = True
    amp_dtype: str = "fp16"
    
    # Checkpointing
    resume: bool = True
    ckpt_keep: int = 5


# ============================================================
# EXAMPLE USAGE AND MODEL SUMMARY
# ============================================================

def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")


def create_model_and_discriminator(cfg: TrainConfigViTGAN):
    """Create generator and discriminator models."""
    
    # Generator (reconstruction model)
    generator = DualDomainMRIReconstructionViTGAN(
        num_cascades=cfg.num_cascades,
        k_base_ch=cfg.k_base_ch,
        k_ffc_blocks=cfg.k_ffc_blocks,
        k_swin_depths=cfg.k_swin_depths,
        k_swin_heads=cfg.k_swin_heads,
        k_window_size=cfg.k_window_size,
        k_coil_chunk=cfg.k_coil_chunk,
        img_base_ch=cfg.img_base_ch,
        img_resblocks=cfg.img_resblocks,
        img_depth=cfg.img_depth,
        img_swin_depths=cfg.img_swin_depths,
        img_swin_heads=cfg.img_swin_heads,
        img_window_size=cfg.img_window_size,
        use_gan_refinement=cfg.use_gan_refinement,
        gan_base_ch=cfg.gan_base_ch,
        gan_residual_blocks=cfg.gan_residual_blocks,
        gan_use_swin=cfg.gan_use_swin,
    )
    
    # Discriminator
    discriminator = MultiScaleDiscriminator(
        in_channels=1,  # Magnitude image
        base_ch=cfg.disc_base_ch,
        n_layers=cfg.disc_n_layers,
        num_discriminators=cfg.num_discriminators,
    )
    
    return generator, discriminator


def demo():
    """Demonstrate model creation and forward pass."""
    print("Creating DDC-KSE-ViT-GAN Model...")
    
    # Configuration
    cfg = TrainConfigViTGAN(
        train_path="./data/train",
        val_path="./data/val",
        num_cascades=4,
        use_gan_refinement=True,
    )
    
    # Create models
    generator, discriminator = create_model_and_discriminator(cfg)
    
    # Print summaries
    print_model_summary(generator, "Generator (DDC-KSE-ViT-GAN)")
    print_model_summary(discriminator, "Multi-Scale Discriminator")
    
    # Demo forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Create dummy input
    B, Nc, H, W = 1, 15, 320, 320
    k_meas = torch.randn(B, Nc, H, W, dtype=torch.complex64, device=device)
    mask = torch.ones(B, 1, H, W, device=device)
    
    print(f"\nDemo forward pass:")
    print(f"  Input k-space: {tuple(k_meas.shape)}")
    print(f"  Mask: {tuple(mask.shape)}")
    
    with torch.no_grad():
        # Generator forward
        recon = generator(k_meas, mask)
        print(f"  Reconstruction: {tuple(recon.shape)}")
        
        # Discriminator forward (on magnitude)
        recon_mag = recon.abs()[:, None]
        disc_out = discriminator(recon_mag)
        print(f"  Discriminator outputs: {[tuple(d.shape) for d in disc_out]}")
    
    print("\n✓ Model creation and forward pass successful!")





# =============================================================================
# INTEGRATED TRAIN / VAL / TEST + VISUALIZATION + TIME CHECKPOINTING (10 min)
# (Merged responsibilities from: mri_train_val_test_integrated.py + train_vit_gan.py)
# Uses the architecture defined above: DDC-KSE-ViT-GAN (Generator + Multi-scale Discriminator)
# =============================================================================

import json
import random
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# ============================================================
# FASTMRI OFFICIAL METRICS (preferred)
# ============================================================

FASTMRI_OFFICIAL = False
try:
    from fastmri.evaluate import nmse as _fmri_nmse, psnr as _fmri_psnr, ssim as _fmri_ssim
    FASTMRI_OFFICIAL = True
    print("[METRICS] Using official fastMRI.evaluate metrics")
except Exception:
    try:
        from fastmri.evaluation.metrics import nmse as _fmri_nmse, psnr as _fmri_psnr, ssim as _fmri_ssim
        FASTMRI_OFFICIAL = True
        print("[METRICS] Using official fastMRI.evaluation.metrics")
    except Exception:
        print("[METRICS] fastMRI official metrics not found (fallback will be used)")

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


class OfficialFastMRIMetrics:
    """
    FastMRI challenge evaluation metrics (NMSE / PSNR / SSIM) with optional center-crop.
    """
    def __init__(self, crop_size: Tuple[int, int] = (320, 320)):
        self.crop_size = crop_size
        self.use_official = FASTMRI_OFFICIAL

    @staticmethod
    def center_crop(x: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        h, w = x.shape[-2:]
        ch, cw = min(crop_size[0], h), min(crop_size[1], w)
        sh = (h - ch) // 2
        sw = (w - cw) // 2
        return x[..., sh:sh + ch, sw:sw + cw]

    def nmse(self, gt: np.ndarray, pred: np.ndarray) -> float:
        if self.use_official:
            return float(_fmri_nmse(gt, pred))
        return float(np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-10))

    def psnr(self, gt: np.ndarray, pred: np.ndarray) -> float:
        if self.use_official:
            try:
                return float(_fmri_psnr(gt, pred, maxval=float(gt.max())))
            except Exception:
                return float(_fmri_psnr(gt, pred))
        mse = float(np.mean((gt - pred) ** 2))
        if mse < 1e-10:
            return 100.0
        return float(20 * np.log10(float(gt.max()) / (math.sqrt(mse) + 1e-12)))

    def ssim(self, gt: np.ndarray, pred: np.ndarray) -> float:
        if self.use_official:
            try:
                return float(_fmri_ssim(gt, pred, maxval=float(gt.max())))
            except Exception:
                try:
                    return float(_fmri_ssim(gt[None], pred[None], maxval=float(gt.max())))
                except Exception:
                    pass
        if SKIMAGE_AVAILABLE:
            return float(skimage_ssim(gt, pred, data_range=float(gt.max() - gt.min())))
        # simple fallback (global)
        mu_x, mu_y = float(np.mean(gt)), float(np.mean(pred))
        sig_x, sig_y = float(np.var(gt)), float(np.var(pred))
        sig_xy = float(np.mean((gt - mu_x) * (pred - mu_y)))
        k1, k2 = 0.01, 0.03
        L = max(float(gt.max()), 1e-6)
        c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
        return float(((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) /
                     ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2) + 1e-12))

    def compute_all(
        self,
        gt: Union[np.ndarray, torch.Tensor],
        pred: Union[np.ndarray, torch.Tensor],
        apply_crop: bool = True,
    ) -> Dict[str, float]:
        if torch.is_tensor(gt):
            if torch.is_complex(gt):
                gt = gt.abs()
            gt = gt.detach().float().cpu().numpy()
        if torch.is_tensor(pred):
            if torch.is_complex(pred):
                pred = pred.abs()
            pred = pred.detach().float().cpu().numpy()

        gt = np.squeeze(gt)
        pred = np.squeeze(pred)

        if apply_crop:
            gt = self.center_crop(gt, self.crop_size)
            pred = self.center_crop(pred, self.crop_size)

        return {'nmse': self.nmse(gt, pred), 'psnr': self.psnr(gt, pred), 'ssim': self.ssim(gt, pred)}


# ============================================================
# BASIC HELPERS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def center_crop_tensor(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    h, w = min(h, H), min(w, W)
    sh, sw = (H - h) // 2, (W - w) // 2
    return x[..., sh:sh + h, sw:sw + w]

import torch.nn.functional as F

def _pad_to_at_least_hw(x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """
    Symmetric zero-pad last two dims to be at least (th, tw).
    Supports [B,H,W], [B,C,H,W], [B,Nc,H,W] and complex tensors.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return x

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad doesn't always behave nicely for complex on some builds -> pad real/imag safely
    if torch.is_complex(x):
        xr = torch.view_as_real(x)  # (..., H, W, 2)
        # move complex-last(2) to channel-like dim for padding
        xr = xr.permute(*range(0, xr.dim()-3), xr.dim()-1, xr.dim()-3, xr.dim()-2)  # (...,2,H,W)
        xr = F.pad(xr, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=0.0)
        xr = xr.permute(*range(0, xr.dim()-3), xr.dim()-2, xr.dim()-1, xr.dim()-3)  # (...,H,W,2)
        return torch.view_as_complex(xr.contiguous())
    else:
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=0.0)


def center_crop_or_pad_tensor(x: torch.Tensor, ch: int, cw: int) -> torch.Tensor:
    """
    Ensure x is at least (ch,cw) by padding, then center-crop to exactly (ch,cw).
    """
    x = _pad_to_at_least_hw(x, ch, cw)
    return center_crop_tensor(x, ch, cw)


def crop_pair(a: torch.Tensor, b: torch.Tensor, h: int, w: int):
    h = min(h, a.shape[-2], b.shape[-2])
    w = min(w, a.shape[-1], b.shape[-1])
    return center_crop_tensor(a, h, w), center_crop_tensor(b, h, w)

def crop_kspace_mask_target(k_meas: torch.Tensor, mask: torch.Tensor, tgt: torch.Tensor, ch: int, cw: int):
    """
    Fix input sizes BEFORE forward:
    - pad to at least (ch,cw)
    - then center crop to (ch,cw)
    """
    # target: [B,H,W] -> [B,ch,cw]
    tgt_c = center_crop_or_pad_tensor(tgt, ch, cw)

    # mask: ensure [B,1,H,W] then pad+crop -> [B,1,ch,cw]
    if mask.dim() == 3:
        mask4 = mask.unsqueeze(1)
    else:
        mask4 = mask
    mask_c = center_crop_or_pad_tensor(mask4, ch, cw)

    # k-space: ifft -> pad+crop in image domain -> fft
    coil_img = ifft2c(k_meas.to(torch.complex64))              # [B,Nc,H,W] complex
    coil_img_c = center_crop_or_pad_tensor(coil_img, ch, cw)   # [B,Nc,ch,cw]
    k_meas_c = fft2c(coil_img_c)                               # [B,Nc,ch,cw] complex

    return k_meas_c, mask_c, tgt_c



def safe_collate(batch):
    """Safe collate function for complex tensors (stacks tensors, keeps strings as lists)."""
    if batch is None or len(batch) == 0:
        return batch
    elem = batch[0]

    if torch.is_tensor(elem):
        return torch.stack([b.contiguous() for b in batch], dim=0)
    if isinstance(elem, np.ndarray):
        return torch.stack([torch.as_tensor(np.ascontiguousarray(b)) for b in batch], dim=0)
    if isinstance(elem, numbers.Number):
        return torch.tensor(batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, dict):
        return {k: safe_collate([d[k] for d in batch]) for k in elem}
    if isinstance(elem, (list, tuple)):
        return [safe_collate(s) for s in zip(*batch)]
    return list(batch)


# ============================================================
# MASKING (Cartesian 1D along readout/W)
# ============================================================

def make_mask_2d(
    H: int,
    W: int,
    acceleration: int,
    center_fraction: float,
    seed: int,
    device=None,
) -> torch.Tensor:
    """
    Create a 2D undersampling mask [1, 1, H, W] for Cartesian sampling (random outside ACS).
    """
    rng = np.random.RandomState(seed)
    num_low = max(1, int(round(W * float(center_fraction))))

    mask = np.zeros(W, dtype=np.float32)
    pad = (W - num_low) // 2
    mask[pad:pad + num_low] = 1.0

    target_cols = max(int(round(W / float(acceleration))), num_low)
    remaining = target_cols - num_low

    if remaining > 0:
        candidates = np.concatenate([np.arange(0, pad), np.arange(pad + num_low, W)])
        if len(candidates) > 0:
            pick = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
            mask[pick] = 1.0

    mask_2d = np.broadcast_to(mask[None, :], (H, W))
    t = torch.from_numpy(mask_2d.copy()).float()[None, None]
    return t.to(device) if device else t


# ============================================================
# DATASET (fastMRI multicoil .h5)
# ============================================================

class FastMRIDataset(Dataset):
    """
    fastMRI multi-coil dataset reader.
    Returns:
      kspace_measured: [Nc, H, W] complex64
      mask: [1, H, W] float32 (0/1)
      target_rss: [H, W] float32 (ground truth magnitude)
      acceleration: int
      fname: str
      slice: int
    """
    def __init__(
        self,
        root: str,
        accelerations: List[int],
        cf_map: Dict[int, float],
        seed: int = 1234,
        max_files: Optional[int] = None,
    ):
        self.root = Path(root)
        self.accelerations = list(accelerations)
        self.cf_map = dict(cf_map)
        self.seed = int(seed)

        assert self.root.exists(), f"Dataset path not found: {self.root}"

        self.files = sorted(self.root.glob("*.h5"))
        if max_files is not None:
            self.files = self.files[:int(max_files)]

        self.examples: List[Tuple[str, int]] = []
        for fp in self.files:
            try:
                with h5py.File(fp, "r") as hf:
                    n_slices = hf["kspace"].shape[0]
                for s in range(n_slices):
                    self.examples.append((str(fp), s))
            except Exception:
                continue

        if len(self.examples) == 0:
            raise RuntimeError(f"No valid examples found in {self.root}")

    def __len__(self):
        return len(self.examples)

    def _pick_acc(self, idx: int) -> int:
        rng = random.Random(self.seed + idx * 17)
        return int(rng.choice(self.accelerations))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fp, sidx = self.examples[idx]
        acc = self._pick_acc(idx)
        cf = float(self.cf_map.get(acc, 0.08))

        with h5py.File(fp, "r") as hf:
            k_full = hf["kspace"][sidx]  # [Nc,H,W] complex64 or complex128
            # Prefer provided RSS if present
            if "reconstruction_rss" in hf:
                target_rss = hf["reconstruction_rss"][sidx]
            else:
                # fallback: RSS from full k-space
                coil_img = ifft2c(torch.as_tensor(k_full))
                target_rss = rss_complex(coil_img[None], dim=1).squeeze(0).cpu().numpy()

        # Normalize k-space (robust)
        k_full_t = torch.as_tensor(k_full)
        if not torch.is_complex(k_full_t):
            k_full_t = torch.view_as_complex(k_full_t)  # if stored as (...,2)
        k_full_t = k_full_t.to(torch.complex64)

        mag = k_full_t.abs()
        scale = torch.quantile(mag.reshape(-1), 0.99).clamp_min(1e-8)
        k_full_t = k_full_t / scale

        # Create mask and apply
        Nc, H, W = k_full_t.shape
        mask = make_mask_2d(H, W, acc, cf, seed=(self.seed + idx), device=None).squeeze(0)  # [1,H,W]
        mask_b = mask.expand(Nc, -1, -1)  # [Nc,H,W]
        k_meas = k_full_t * mask_b.to(k_full_t.device)

        # Target magnitude (normalize consistent with kspace scale)
        target_rss_t = torch.as_tensor(np.ascontiguousarray(target_rss)).float()
        target_rss_t = target_rss_t / (float(scale.cpu()) + 1e-12)

        return {
            "kspace_measured": k_meas.contiguous(),
            "mask": mask.contiguous(),
            "target_rss": target_rss_t.contiguous(),
            "acceleration": torch.tensor(acc, dtype=torch.int32),
            "fname": Path(fp).stem,
            "slice": torch.tensor(sidx, dtype=torch.int32),
        }


# ============================================================
# COIL-GROUPED BATCH SAMPLER
# Ensures every batch has identical Nc — no zero-padding waste.
# ============================================================

class CoilGroupedBatchSampler(Sampler):
    def __init__(self, dataset: FastMRIDataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Read coil count once per FILE — all slices share the same Nc.
        nc_per_file: Dict[str, int] = {}
        for fp, _ in dataset.examples:
            if fp not in nc_per_file:
                try:
                    with h5py.File(fp, "r") as hf:
                        nc_per_file[fp] = int(hf["kspace"].shape[1])
                except Exception:
                    nc_per_file[fp] = -1

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, (fp, _) in enumerate(dataset.examples):
            nc = nc_per_file.get(fp, -1)
            if nc > 0:
                groups[nc].append(idx)

        # Build complete batches per group — drop last incomplete batch
        self.batches: List[List[int]] = []
        for nc, idxs in groups.items():
            for start in range(0, len(idxs) - batch_size + 1, batch_size):
                self.batches.append(idxs[start:start + batch_size])

        print(f"[CoilGroupedBatchSampler] coil groups: { {nc: len(idxs) for nc, idxs in groups.items()} }")
        print(f"[CoilGroupedBatchSampler] total batches: {len(self.batches)} (batch_size={batch_size})")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
            for b in batches:
                rng.shuffle(b)
        for b in batches:
            yield b

    def __len__(self):
        return len(self.batches)


# ============================================================
# VISUALIZER (5-panel: Recon | Masked(ZF) | GT | Error | RelError)
# ============================================================

class AcademicVisualizer5:
    """
    Publication-quality 5-panel figure every N steps:
    (reconstruction, masked image, ground truth, error, relative error)
    """
    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)

        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 12,
        })

    def _safe_name(self, s: str) -> str:
        return Path(str(s)).stem.replace(" ", "_").replace(":", "_")[:90]

    def save_figure(
            self,
            recon: torch.Tensor,
            masked: torch.Tensor,
            gt: torch.Tensor,
            metrics: Dict[str, float],
            filename: str,
            title: str = "",
            sidx: Optional[int] = None,
            acceleration: int = 4,
            eps: float = 1e-8,
    ) -> Path:
        recon_np = recon.squeeze().detach().float().cpu().numpy()
        masked_np = masked.squeeze().detach().float().cpu().numpy()
        gt_np = gt.squeeze().detach().float().cpu().numpy()

        err = np.abs(gt_np - recon_np)
        rel = err / (np.abs(gt_np) + eps)

        vmax = float(np.percentile(gt_np, 99.5))
        vmin = 0.0

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # 1) Reconstruction
        axes[0].imshow(recon_np, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Recon\nPSNR={metrics.get('psnr',0):.2f}dB")
        axes[0].axis("off")

        # 2) Masked image (ZF)
        axes[1].imshow(masked_np, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title("Masked (ZF)")
        axes[1].axis("off")

        # 3) Ground truth
        axes[2].imshow(gt_np, cmap="gray", vmin=vmin, vmax=vmax)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # 4) Error
        im1 = axes[3].imshow(err, cmap="hot")
        axes[3].set_title(f"Error\nSSIM(recon,gt)={metrics.get('ssim',0):.4f}")
        axes[3].axis("off")
        div1 = make_axes_locatable(axes[3])
        cax1 = div1.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(im1, cax=cax1)

        # 5) Relative error
        im2 = axes[4].imshow(rel, cmap="viridis")
        axes[4].set_title(f"Rel. Error\nNMSE={metrics.get('nmse',0):.4f}")
        axes[4].axis("off")
        div2 = make_axes_locatable(axes[4])
        cax2 = div2.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(im2, cax=cax2)

        if title:
            if sidx is not None:
                title = f"{title} | slice={int(sidx)}"
            fig.suptitle(f"{title} | {acceleration}x", fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / f"{self._safe_name(filename)}.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path


# ============================================================
# BEST / WORST TRACKER
# ============================================================

class BestWorstTracker:
    """Track best/worst samples by PSNR/SSIM per acceleration."""
    def __init__(self, accelerations: List[int]):
        self.accelerations = accelerations
        self.best = {acc: {"psnr": None, "ssim": None} for acc in accelerations}
        self.worst = {acc: {"psnr": None, "ssim": None} for acc in accelerations}

    def update(self, acc: int, gt: torch.Tensor, pred: torch.Tensor, zf: torch.Tensor, metrics: Dict[str, float], fname: str = ""):
        sample = {"gt": gt.detach().cpu(), "pred": pred.detach().cpu(), "zf": zf.detach().cpu(), "metrics": dict(metrics), "fname": fname}
        psnr = float(metrics.get("psnr", 0.0))
        ssim = float(metrics.get("ssim", 0.0))

        if self.best[acc]["psnr"] is None or psnr > float(self.best[acc]["psnr"]["metrics"]["psnr"]):
            self.best[acc]["psnr"] = sample
        if self.worst[acc]["psnr"] is None or psnr < float(self.worst[acc]["psnr"]["metrics"]["psnr"]):
            self.worst[acc]["psnr"] = sample

        if self.best[acc]["ssim"] is None or ssim > float(self.best[acc]["ssim"]["metrics"]["ssim"]):
            self.best[acc]["ssim"] = sample
        if self.worst[acc]["ssim"] is None or ssim < float(self.worst[acc]["ssim"]["metrics"]["ssim"]):
            self.worst[acc]["ssim"] = sample


# ============================================================
# CHECKPOINT MANAGER (Generator + Discriminator)
# ============================================================

class GANCheckpointManager:
    """Checkpoint manager for GAN training (G/D + optimizers + scalers)."""

    def __init__(self, ckpt_dir: str, name: str = "vit_gan", max_keep: int = 5, monitor: str = "psnr"):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.max_keep = int(max_keep)
        self.monitor = monitor
        self.best = -float("inf") if monitor in ["psnr", "ssim"] else float("inf")

    def save(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        scaler_g: Optional[GradScaler],
        scaler_d: Optional[GradScaler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, bool]:
        cur = float(metrics.get(self.monitor, self.best))
        is_better = cur > self.best if self.monitor in ["psnr", "ssim"] else cur < self.best
        if is_better:
            self.best = cur

        payload = {
            "epoch": int(epoch),
            "step": int(step),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "scaler_g": scaler_g.state_dict() if scaler_g else None,
            "scaler_d": scaler_d.state_dict() if scaler_d else None,
            "metrics": dict(metrics),
            "best": float(self.best),
            "extra": extra or {},
        }

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = self.dir / f"{self.name}_e{int(epoch):04d}_{ts}.pt"
        torch.save(payload, ckpt_path)

        last_path = self.dir / f"{self.name}_last.pt"
        torch.save(payload, last_path)

        if is_better:
            best_path = self.dir / f"{self.name}_best.pt"
            torch.save(payload, best_path)
            print(f"  ★ New best {self.monitor}={cur:.4f}")

        # cleanup
        ckpts = sorted(self.dir.glob(f"{self.name}_e*.pt"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.max_keep:
            ckpts.pop(0).unlink(missing_ok=True)

        return ckpt_path, is_better

    def load(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        optim_g: Optional[torch.optim.Optimizer] = None,
        optim_d: Optional[torch.optim.Optimizer] = None,
        scaler_g: Optional[GradScaler] = None,
        scaler_d: Optional[GradScaler] = None,
        best: bool = True,
        device: str = "cuda",
        strict: bool = False,
     ) -> Dict[str, Any]:
        path = self.dir / f"{self.name}_{'best' if best else 'last'}.pt"
        if not path.exists():
            return {"epoch": 0, "step": 0}
        print(f"[CKPT] Loading {path}")
        ckpt = torch.load(path, map_location=device)

        generator.load_state_dict(ckpt.get("generator", {}), strict=strict)
        discriminator.load_state_dict(ckpt.get("discriminator", {}), strict=strict)

        if optim_g and "optim_g" in ckpt:
            try:
                optim_g.load_state_dict(ckpt["optim_g"])
            except Exception:
                pass
        if optim_d and "optim_d" in ckpt:
            try:
                optim_d.load_state_dict(ckpt["optim_d"])
            except Exception:
                pass
        if scaler_g and ckpt.get("scaler_g") is not None:
            try:
                scaler_g.load_state_dict(ckpt["scaler_g"])
            except Exception:
                pass
        if scaler_d and ckpt.get("scaler_d") is not None:
            try:
                scaler_d.load_state_dict(ckpt["scaler_d"])
            except Exception:
                pass

        self.best = float(ckpt.get("best", self.best))
        return {"epoch": int(ckpt.get("epoch", 0)), "step": int(ckpt.get("step", 0))}



# ============================================================
# PER-ITERATION LOGGER (resume-safe, immediate flush)
# ============================================================

class IterationLogger:
    """
    Writes one CSV row per training iteration to:
        <out_dir>/logs/train_iter_log.csv

    Columns:
        epoch, iteration, global_step, acceleration,
        psnr, ssim, nmse,
        loss_total, loss_l1, loss_ssim, loss_perceptual, loss_adversarial, loss_d

    Resume-safe:
        - Fresh run  (resume_from_step=0)  → creates new file with header
        - Resumed run (resume_from_step=N) → appends to existing file,
          inserts a timestamp separator, skips steps already logged

    flush_every=1 (default) → every row written to disk immediately.
    No data lost if training crashes.
    """

    HEADER = (
        "epoch,iteration,global_step,acceleration,"
        "psnr,ssim,nmse,"
        "loss_total,loss_l1,loss_ssim,loss_perceptual,loss_adversarial,loss_d\n"
    )

    def __init__(self, out_dir, flush_every: int = 1, resume_from_step: int = 0):
        import datetime
        from pathlib import Path
        log_dir = Path(out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._path         = log_dir / "train_iter_log.csv"
        self._flush_every  = max(1, flush_every)
        self._buffer: list = []
        self._total_rows   = 0
        self._resume_step  = resume_from_step

        is_resume = (resume_from_step > 0) and self._path.exists()

        if is_resume:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(f"# ── RESUMED from step {resume_from_step} at {ts} ──\n")
            print(
                f"[IterationLogger] RESUME → appending to {self._path} "
                f"(previous data preserved, continuing from step {resume_from_step})"
            )
        else:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(self.HEADER)
            print(f"[IterationLogger] NEW run → {self._path}")

    def log(self, epoch, iteration, global_step, acceleration,
            psnr, ssim, nmse, g_losses, loss_d=0.0):
        """
        Call once per iteration BEFORE self.global_step += 1.

        Skip logic (prevents duplicates on resume):
          - global_step < resume_from_step  → already logged, skip
          - global_step == resume_from_step AND resume_from_step > 0 → last
            step before crash was already written, skip
        """
        if global_step < self._resume_step:
            return
        if global_step == self._resume_step and self._resume_step > 0:
            return

        row = (
            f"{epoch},{iteration},{global_step},{acceleration},"
            f"{psnr:.4f},{ssim:.4f},{nmse:.6f},"
            f"{g_losses.get('total',       0.0):.6f},"
            f"{g_losses.get('l1',          0.0):.6f},"
            f"{g_losses.get('ssim',        0.0):.6f},"
            f"{g_losses.get('perceptual',  0.0):.6f},"
            f"{g_losses.get('adversarial', 0.0):.6f},"
            f"{loss_d:.6f}\n"
        )
        self._buffer.append(row)
        self._total_rows += 1
        if len(self._buffer) >= self._flush_every:
            self._flush()

    def close(self):
        """Flush remaining rows. Call at end of training."""
        self._flush()
        print(f"[IterationLogger] Closed — {self._total_rows:,} rows written to {self._path}")

    def _flush(self):
        if not self._buffer:
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.writelines(self._buffer)
        except Exception as e:
            print(f"[IterationLogger] WARNING: could not write: {e}")
        self._buffer.clear()

# ============================================================
# UNIFIED TRAINER (train / validate / test)
# ============================================================

@dataclass
class TrainConfigAllInOne:
    # Paths (prefilled; override via CLI)
    train_path: str = os.environ.get("FASTMRI_TRAIN", r"D:\train\multicoil_train")
    val_path: str = os.environ.get("FASTMRI_VAL", r"D:\train\multicoil_val")
    test_path: str = os.environ.get("FASTMRI_TEST", r"D:\train\multicoil_test_full")
    out_dir: str = "./outputs_vit_gan_integrated"

    # Data
    batch_size: int = 1
    num_workers: int = 4
    accelerations: Tuple[int, ...] = (4, 8)
    cf_map: Dict[int, float] = field(default_factory=lambda: {4: 0.08, 8: 0.04})
    train_crop_hw: Tuple[int, int] = (320, 320)  # training crop (loss/metrics/vis)
    eval_crop_hw: Tuple[int, int] = (320, 320)   # evaluation crop (paper reporting)
    eval_average: str = "slice"                 # "slice" or "volume" averaging in eval
    max_files: Optional[int] = None  # for quick debug

    # Model (Generator)
    num_cascades: int = 8
    k_base_ch: int = 32
    k_ffc_blocks: int = 2
    k_swin_depths: Tuple[int, ...] = (1, 1, 1)
    k_swin_heads: Tuple[int, ...] = (2, 4, 8)
    k_window_size: int = 8
    k_coil_chunk: int = 1
    img_base_ch: int = 32
    img_resblocks: int = 1
    img_depth: int = 2
    img_swin_depths: Tuple[int, ...] = (1, 1, 1)
    img_swin_heads: Tuple[int, ...] = (2, 4, 8)
    img_window_size: int = 8

    # Model (GAN)
    use_gan_refinement: bool = True
    gan_start_epoch: int = 5
    d_steps_per_g: int = 1
    gan_base_ch: int = 64
    gan_residual_blocks: int = 6
    gan_use_swin: bool = True
    disc_base_ch: int = 64
    disc_n_layers: int = 3
    num_discriminators: int = 2

    # Training
    epochs: int = 50
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    grad_clip: float = 1.0

    empty_cache_every: int = 50  # torch.cuda.empty_cache() every N steps (0=off)
    skip_oom: bool = True        # skip batch on CUDA OOM (set --no_skip_oom to crash)

    # Loss weights
    l1_weight: float = 1.0
    ssim_weight: float = 0.1
    perceptual_weight: float = 0.1
    adversarial_weight: float = 0.01

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # Visualization
    vis_every: int = 10
    max_vis_per_epoch: int = 50
    save_best_worst: bool = True

    # Checkpointing
    resume: bool = True
    resume_best: bool = False
    ckpt_keep: int = 5
    time_ckpt_minutes: int = 10  # requested: every 10 minutes

    # Logging
    print_every: int = 400
    tb_every: int = 1   # log every step to TensorBoard
    seed: int = 42


class TrainerViTGANIntegrated:
    """
    Train / Validate / Test trainer for DDC-KSE-ViT-GAN with:
      - figure every cfg.vis_every steps (Recon | Masked(ZF) | GT | Error | RelError)
      - time checkpoint every cfg.time_ckpt_minutes (if >0)
      - per-epoch best/last checkpoints
      - official fastMRI metrics (if available)
    """
    def __init__(self, cfg: TrainConfigAllInOne, device: Optional[Union[str, torch.device]] = None):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # perf flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")  # pytorch 2.x
        except Exception:
            pass

        # Output dirs
        self.out = Path(cfg.out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "visualizations" / "train").mkdir(parents=True, exist_ok=True)
        (self.out / "visualizations" / "val").mkdir(parents=True, exist_ok=True)
        (self.out / "visualizations" / "test").mkdir(parents=True, exist_ok=True)
        (self.out / "visualizations" / "best_worst").mkdir(parents=True, exist_ok=True)
        (self.out / "results").mkdir(parents=True, exist_ok=True)
        (self.out / "tb").mkdir(parents=True, exist_ok=True)

        # Metrics + Visualizers
        self.metrics = OfficialFastMRIMetrics(crop_size=cfg.eval_crop_hw)
        self.train_vis = AcademicVisualizer5(str(self.out / "visualizations" / "train"), dpi=600)
        self.val_vis = AcademicVisualizer5(str(self.out / "visualizations" / "val"), dpi=600)
        self.test_vis = AcademicVisualizer5(str(self.out / "visualizations" / "test"), dpi=600)
        self.bw_vis = AcademicVisualizer5(str(self.out / "visualizations" / "best_worst"), dpi=600)
        self.tracker = BestWorstTracker(list(cfg.accelerations)) if cfg.save_best_worst else None

        # Models
        self.generator, self.discriminator = create_model_and_discriminator(cfg)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Optimizers
        self.optim_g = torch.optim.AdamW(self.generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-5)
        self.optim_d = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-5)

        # AMP
        self.scaler_g = GradScaler(enabled=(cfg.use_amp and self.device.type == "cuda"))
        self.scaler_d = GradScaler(enabled=(cfg.use_amp and self.device.type == "cuda"))

        # Losses
        self.criterion_recon = CombinedLoss(
            l1_weight=cfg.l1_weight,
            ssim_weight=cfg.ssim_weight,
            perceptual_weight=0.0,
            adversarial_weight=0.0,
            use_perceptual=False,
        )
        self.criterion_gan = CombinedLoss(
            l1_weight=cfg.l1_weight,
            ssim_weight=cfg.ssim_weight,
            perceptual_weight=cfg.perceptual_weight,
            adversarial_weight=cfg.adversarial_weight,
            use_perceptual=(cfg.perceptual_weight > 0),
        )
        self.gan_loss = GANLoss(gan_mode="lsgan")

        # IMPORTANT: loss modules contain buffers / subnets (e.g., VGG perceptual) that must be on the same device
        # as the training tensors. Otherwise you will hit CPU/CUDA mismatch errors.
        self.criterion_recon = self.criterion_recon.to(self.device)
        self.criterion_gan = self.criterion_gan.to(self.device)
        self.gan_loss = self.gan_loss.to(self.device)

        # Checkpoints
        self.ckpt_mgr = GANCheckpointManager(str(self.out / "checkpoints"), name="vit_gan", max_keep=cfg.ckpt_keep, monitor="psnr")

        # TensorBoard
        self.writer = SummaryWriter(str(self.out / "tb")) if TENSORBOARD_AVAILABLE else None

        # State
        self.start_epoch = 0
        self.global_step = 0
        self._vis_count = 0
        self._last_time_ckpt = time.time()

        # Resume
        if cfg.resume:
            st = self.ckpt_mgr.load(
                self.generator, self.discriminator,
                self.optim_g, self.optim_d,
                self.scaler_g, self.scaler_d,
                best=cfg.resume_best,
                device=str(self.device),
                strict=False,
            )
            self.start_epoch = int(st.get("epoch", 0))
            self.global_step = int(st.get("step", 0))

        # ── Per-iteration CSV logger ──────────────────────────────────
        # global_step is already restored from checkpoint above.
        # IterationLogger uses it to decide append vs fresh + skip duplicates.
        self.iter_logger = IterationLogger(
            out_dir          = self.out,
            flush_every      = 1,   # write every row immediately — no data lost on crash
            resume_from_step = self.global_step,
        )

        n_g = sum(p.numel() for p in self.generator.parameters())
        n_d = sum(p.numel() for p in self.discriminator.parameters())
        print(f"[DEVICE] {self.device}")
        print(f"[MODEL] Generator params: {n_g:,} | Discriminator params: {n_d:,}")
        print(f"[CONFIG] vis_every={cfg.vis_every}, time_ckpt_minutes={cfg.time_ckpt_minutes}, accelerations={cfg.accelerations}, train_crop={cfg.train_crop_hw}, eval_crop={cfg.eval_crop_hw}, eval_average={cfg.eval_average}")

    def _amp_dtype(self):
        return torch.bfloat16 if str(self.cfg.amp_dtype).lower() == "bf16" else torch.float16

    def _compute_zf(self, k_meas: torch.Tensor) -> torch.Tensor:
        """Zero-filled (masked) image from measured k-space."""
        if k_meas.dim() == 3:
            k_meas = k_meas.unsqueeze(0)  # [1,Nc,H,W]
        coil_img = ifft2c(k_meas.to(torch.complex64))
        zf = rss_complex(coil_img, dim=1).float()  # [B,H,W]
        return zf

    def _prep_batch(self, batch: Dict[str, Any]):
        k_meas = batch["kspace_measured"].to(self.device)
        mask = batch["mask"].to(self.device)
        tgt = batch["target_rss"].to(self.device)

        if k_meas.dim() == 3:
            k_meas = k_meas.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)  # [B,1,H,W]

        acc = int(batch["acceleration"][0].item()) if torch.is_tensor(batch["acceleration"]) else int(batch["acceleration"])
        fname = batch["fname"][0] if isinstance(batch["fname"], list) else str(batch["fname"])

        return k_meas, mask, tgt, acc, fname

    def _forward_and_crop(self, k_meas: torch.Tensor, mask: torch.Tensor, tgt: torch.Tensor,
                          crop_hw: Optional[Tuple[int, int]] = None):

        ch, cw = (crop_hw if crop_hw is not None else self.cfg.eval_crop_hw)

        # >>> CROP BEFORE FORWARD <<<
        k_meas_c, mask_c, tgt_c = crop_kspace_mask_target(k_meas, mask, tgt, ch, cw)

        amp_on = self.cfg.use_amp and self.device.type == "cuda"
        with autocast(enabled=amp_on, dtype=self._amp_dtype()):
            recon = self.generator(k_meas_c, mask_c)  # complex [B,ch,cw]
            recon_mag = recon.abs().float()  # [B,ch,cw]

        # Normalize for loss / discriminator stability
        tgt_max = tgt_c.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        tgt_n = tgt_c / tgt_max
        pred_n = (recon_mag / tgt_max).clamp(0, 2)

        # pred_crop = recon_mag artık zaten crop’lı
        return recon_mag, recon_mag, tgt_c, pred_n, tgt_n

    def _train_discriminator(self, pred_n: torch.Tensor, tgt_n: torch.Tensor) -> Dict[str, float]:
        self.discriminator.train()
        self.generator.eval()

        amp_on = self.cfg.use_amp and self.device.type == "cuda"

        with autocast(enabled=amp_on, dtype=self._amp_dtype()):
            disc_real = self.discriminator(tgt_n[:, None])
            disc_fake = self.discriminator(pred_n.detach()[:, None])

            loss_real = 0.0
            for d_real in disc_real:
                loss_real = loss_real + self.gan_loss(d_real, True)
            loss_real = loss_real / max(len(disc_real), 1)

            loss_fake = 0.0
            for d_fake in disc_fake:
                loss_fake = loss_fake + self.gan_loss(d_fake, False)
            loss_fake = loss_fake / max(len(disc_fake), 1)

            loss_d = 0.5 * (loss_real + loss_fake)

        self.optim_d.zero_grad(set_to_none=True)
        self.scaler_d.scale(loss_d).backward()
        if self.cfg.grad_clip > 0:
            self.scaler_d.unscale_(self.optim_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.grad_clip)
        self.scaler_d.step(self.optim_d)
        self.scaler_d.update()

        return {"loss_d": float(loss_d.item()), "loss_d_real": float(loss_real.item()), "loss_d_fake": float(loss_fake.item())}

    def _train_generator(self, pred_n: torch.Tensor, tgt_n: torch.Tensor, use_gan: bool) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.eval()

        amp_on = self.cfg.use_amp and self.device.type == "cuda"
        with autocast(enabled=amp_on, dtype=self._amp_dtype()):
            if use_gan:
                disc_fake_for_g = self.discriminator(pred_n[:, None])
                losses = self.criterion_gan(pred_n, tgt_n, disc_fake_for_g)
            else:
                losses = self.criterion_recon(pred_n, tgt_n, None)

        self.optim_g.zero_grad(set_to_none=True)
        self.scaler_g.scale(losses["total"]).backward()
        if self.cfg.grad_clip > 0:
            self.scaler_g.unscale_(self.optim_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.cfg.grad_clip)
        self.scaler_g.step(self.optim_g)
        self.scaler_g.update()

        return {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        if hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)

        use_gan = (self.cfg.use_gan_refinement and epoch > self.cfg.gan_start_epoch)

        sums = defaultdict(float)
        n = 0
        self._vis_count = 0

        for it, batch in enumerate(loader):
            k_meas, mask, tgt, acc, fname = self._prep_batch(batch)
            # raise RuntimeError("shape mismatch")
            # Forward / crop
            pred_n = None
            tgt_n = None
            try:
                recon_mag, pred_crop, tgt_crop, pred_n, tgt_n = self._forward_and_crop(
                    k_meas, mask, tgt, crop_hw=self.cfg.train_crop_hw
                )
                # ✅ Shape check MUST be here (after assignment)
                if pred_n.shape != tgt_n.shape:
                    print(f"[SHAPE MISMATCH] pred_n={tuple(pred_n.shape)} tgt_n={tuple(tgt_n.shape)}")
                    # İstersen batch'i atla:
                    continue
                    # veya raise:
                    # raise RuntimeError("Shape mismatch after _forward_and_crop")
            except RuntimeError as e:
                msg = str(e).lower()
                if self.cfg.skip_oom and ("out of memory" in msg or "cuda error" in msg):
                    print(f"  [OOM] Skipping batch e{epoch} it{it} (FORWARD). Cleaning up.")
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

            # Train D (optional)
            if use_gan:
                d_sums = {"loss_d": 0.0}
                for _ in range(max(1, int(self.cfg.d_steps_per_g))):
                    d_losses = self._train_discriminator(pred_n, tgt_n)
                    d_sums["loss_d"] += d_losses["loss_d"]
                sums["loss_d"] += d_sums["loss_d"] / max(1, int(self.cfg.d_steps_per_g))

            # Train G
            try:
                g_losses = self._train_generator(pred_n, tgt_n, use_gan)
            except RuntimeError as e:
                msg = str(e).lower()
                if self.cfg.skip_oom and ("out of memory" in msg or "cuda error" in msg):
                    print(f"  [OOM] Skipping batch e{epoch} it{it} (G backward). Cleaning up.")
                    # Drop grads (and any partially-built graphs)
                    try:
                        self.optim_g.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    if use_gan:
                        try:
                            self.optim_d.zero_grad(set_to_none=True)
                        except Exception:
                            pass

                    # Drop references to large tensors ASAP
                    for _nm in ("recon_mag", "pred_crop", "tgt_crop", "pred_n", "tgt_n"):
                        try:
                            del locals()[_nm]  # may not exist depending on failure point
                        except Exception:
                            pass

                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass

                    if self.device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        except Exception:
                            pass
                    continue
                raise

            # Optional periodic cache clear to reduce fragmentation
            if self.device.type == "cuda" and self.cfg.empty_cache_every and self.cfg.empty_cache_every > 0:
                if ((it + 1) % int(self.cfg.empty_cache_every)) == 0:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            # Metrics (on unnormalized cropped mags)
            with torch.no_grad():
                met = self.metrics.compute_all(tgt_crop, pred_crop)
                zf = self._compute_zf(k_meas)
                zf_crop = center_crop_tensor(zf, *self.cfg.train_crop_hw)

            n += 1
            sums["loss"] += g_losses["total"]
            sums["l1"] += g_losses.get("l1", 0.0)
            sums["ssim_loss"] += g_losses.get("ssim", 0.0)
            sums["perceptual"] += g_losses.get("perceptual", 0.0)
            sums["adversarial"] += g_losses.get("adversarial", 0.0)
            sums["psnr"] += met["psnr"]
            sums["ssim"] += met["ssim"]
            sums["nmse"] += met["nmse"]

            # ── Per-iteration log ──────────────────────────────────────
            self.iter_logger.log(
                epoch        = epoch,
                iteration    = it,
                global_step  = self.global_step,
                acceleration = acc,
                psnr         = met["psnr"],
                ssim         = met["ssim"],
                nmse         = met["nmse"],
                g_losses     = g_losses,
                loss_d       = sums.get("loss_d", 0.0) / max(n, 1),
            )

            self.global_step += 1

            # ===== FIGURE EVERY vis_every STEPS =====
            _slice = batch.get("slice", 0)
            if isinstance(_slice, (list, tuple)):
                _slice = _slice[0]
            if torch.is_tensor(_slice):
                sidx = int(_slice.flatten()[0].item())
            else:
                sidx = int(_slice)
            if (self.global_step % self.cfg.vis_every == 0) and (self._vis_count < self.cfg.max_vis_per_epoch):
                self.train_vis.save_figure(
                    recon=pred_crop[0],
                    masked=zf_crop[0],
                    gt=tgt_crop[0],
                    metrics=met,
                    sidx=sidx,
                    filename=f"train_e{epoch:03d}_it{it:05d}_s{self.global_step:07d}_sl{sidx:04d}_{fname[:20]}",
                    title=f"Train Step {self.global_step}",
                    acceleration=acc,
                )
                self._vis_count += 1

            # Print
            if it % self.cfg.print_every == 0:
                mode = "GAN" if use_gan else "RECON"
                sidx = int(batch["slice"][0].item()) if torch.is_tensor(batch["slice"]) else int(batch["slice"])
                print(
                    f"  [train-{mode}] e{epoch} it{it}/{len(loader)} step={self.global_step} acc={acc} "
                    f"fname={fname} slice={sidx} "
                    f"loss={g_losses['total']:.4f} PSNR={met['psnr']:.2f} SSIM={met['ssim']:.4f} NMSE={met['nmse']:.4f}"
                )
            # TensorBoard — every step, all metrics
            if self.writer:
                self.writer.add_scalar("train/loss",             g_losses["total"],                    self.global_step)
                self.writer.add_scalar("train/loss_l1",          g_losses.get("l1",          0.0),     self.global_step)
                self.writer.add_scalar("train/loss_ssim",        g_losses.get("ssim",        0.0),     self.global_step)
                self.writer.add_scalar("train/loss_perceptual",  g_losses.get("perceptual",  0.0),     self.global_step)
                self.writer.add_scalar("train/loss_adversarial", g_losses.get("adversarial", 0.0),     self.global_step)
                self.writer.add_scalar("train/psnr",             met["psnr"],                          self.global_step)
                self.writer.add_scalar("train/ssim",             met["ssim"],                          self.global_step)
                self.writer.add_scalar("train/nmse",             met["nmse"],                          self.global_step)
                if use_gan:
                    self.writer.add_scalar("train/loss_d",       sums.get("loss_d", 0.0) / max(n, 1), self.global_step)
                    self.writer.add_scalar("train/loss_d_real",  sums.get("loss_d_real", 0.0) / max(n, 1), self.global_step)
                    self.writer.add_scalar("train/loss_d_fake",  sums.get("loss_d_fake", 0.0) / max(n, 1), self.global_step)

            # ===== TIME CHECKPOINT (10 minutes default) =====
            if self.cfg.time_ckpt_minutes and self.cfg.time_ckpt_minutes > 0:
                if (time.time() - self._last_time_ckpt) > float(self.cfg.time_ckpt_minutes) * 60.0:
                    self._last_time_ckpt = time.time()
                    avg = {k: v / max(n, 1) for k, v in sums.items()}
                    self.ckpt_mgr.save(
                        self.generator, self.discriminator,
                        self.optim_g, self.optim_d,
                        self.scaler_g, self.scaler_d,
                        epoch=epoch, step=self.global_step,
                        metrics={"psnr": avg.get("psnr", 0.0), "ssim": avg.get("ssim", 0.0), "nmse": avg.get("nmse", 0.0)},
                        extra={"time_ckpt": True, "epoch_avg": avg, "use_gan": use_gan},
                    )
                    print("  [CKPT] Time-based checkpoint saved")

        return {k: v / max(n, 1) for k, v in sums.items()}
    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int, split: str = "val") -> Dict[str, float]:
        """Validate/Test with either slice- or volume-averaged fastMRI metrics.

        cfg.eval_average:
          - "slice": mean over all slices in the loader (common for quick dev)
          - "volume": mean over volumes (fname), where each volume metric is the mean of its slices
        """
        self.generator.eval()
        vis = self.val_vis if split == "val" else self.test_vis

        eval_crop = tuple(self.cfg.eval_crop_hw)
        average = str(getattr(self.cfg, "eval_average", "slice")).lower()
        if average not in ("slice", "volume"):
            average = "slice"

        # Slice-level accumulators (overall)
        sums = defaultdict(float)
        zf_sums = defaultdict(float)
        all_psnr, all_ssim = [], []
        n_slices = 0

        # Volume identifiers (acc, fname) so mixed-acc loaders remain correct
        vol_names = set()

        # Per-acc slice-level accumulators (used when average=="slice")
        slice_sums_by_acc = defaultdict(lambda: defaultdict(float))
        zf_slice_sums_by_acc = defaultdict(lambda: defaultdict(float))
        all_psnr_by_acc = defaultdict(list)
        all_ssim_by_acc = defaultdict(list)
        n_slices_by_acc = defaultdict(int)

        # Volume-level buffers
        # - overall: key=(acc,fname) -> list[metric dict]
        # - by-acc:  acc -> fname -> list[metric dict]
        vol_mets = defaultdict(list)
        vol_zf_mets = defaultdict(list)
        vol_mets_by_acc = defaultdict(lambda: defaultdict(list))
        vol_zf_mets_by_acc = defaultdict(lambda: defaultdict(list))

        for it, batch in enumerate(loader):
            k_meas, mask, tgt, acc, fname = self._prep_batch(batch)
            vol_names.add((acc, fname))

            pred_n = None
            tgt_n = None
            try:
                recon_mag, pred_crop, tgt_crop, pred_n, tgt_n = self._forward_and_crop(
                    k_meas, mask, tgt, crop_hw=self.cfg.train_crop_hw
                )
                # ✅ Shape check MUST be here (after assignment)
                if pred_n.shape != tgt_n.shape:
                    print(f"[SHAPE MISMATCH] pred_n={tuple(pred_n.shape)} tgt_n={tuple(tgt_n.shape)}")
                    # İstersen batch'i atla:
                    continue
                    # veya raise:
                    # raise RuntimeError("Shape mismatch after _forward_and_crop")
            except RuntimeError as e:
                msg = str(e).lower()
                if self.cfg.skip_oom and ("out of memory" in msg or "cuda error" in msg):
                    print(f"  [OOM] Skipping batch e{epoch} it{it} (FORWARD). Cleaning up.")
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

            # Zero-filled baseline
            zf = self._compute_zf(k_meas)

            # IMPORTANT:
            # fastMRI official metrics expect gt and pred arrays to have identical shapes.
            # Some volumes/slices can have slightly different in-plane sizes (e.g., 320x260 vs 320x262)
            # between target and ZF (and occasionally model output depending on padding).
            # To avoid broadcasting errors and to keep evaluation consistent, align ALL tensors
            # (pred, gt, zf) to the common center-cropped intersection size before metrics.
            h_final = min(int(pred_crop.shape[-2]), int(tgt_crop.shape[-2]), int(zf.shape[-2]))
            w_final = min(int(pred_crop.shape[-1]), int(tgt_crop.shape[-1]), int(zf.shape[-1]))

            if (int(pred_crop.shape[-2]), int(pred_crop.shape[-1])) != (h_final, w_final):
                pred_crop = center_crop_tensor(pred_crop, h_final, w_final)
            if (int(tgt_crop.shape[-2]), int(tgt_crop.shape[-1])) != (h_final, w_final):
                tgt_crop = center_crop_tensor(tgt_crop, h_final, w_final)
            zf_crop = center_crop_tensor(zf, h_final, w_final)

            met = self.metrics.compute_all(tgt_crop, pred_crop)
            zf_met = self.metrics.compute_all(tgt_crop, zf_crop)

            n_slices += 1

            # Always keep buffers for volume averaging and per-acc summaries
            vol_key = (acc, fname)
            vol_mets[vol_key].append(met)
            vol_zf_mets[vol_key].append(zf_met)
            vol_mets_by_acc[acc][fname].append(met)
            vol_zf_mets_by_acc[acc][fname].append(zf_met)

            # Slice-level accumulators (overall)
            sums["psnr"] += met["psnr"]
            sums["ssim"] += met["ssim"]
            sums["nmse"] += met["nmse"]
            all_psnr.append(met["psnr"])
            all_ssim.append(met["ssim"])

            zf_sums["psnr"] += zf_met["psnr"]
            zf_sums["ssim"] += zf_met["ssim"]
            zf_sums["nmse"] += zf_met["nmse"]

            # Slice-level per-acc accumulators (for by-acc in slice mode)
            n_slices_by_acc[acc] += 1
            slice_sums_by_acc[acc]["psnr"] += met["psnr"]
            slice_sums_by_acc[acc]["ssim"] += met["ssim"]
            slice_sums_by_acc[acc]["nmse"] += met["nmse"]
            all_psnr_by_acc[acc].append(met["psnr"])
            all_ssim_by_acc[acc].append(met["ssim"])

            zf_slice_sums_by_acc[acc]["psnr"] += zf_met["psnr"]
            zf_slice_sums_by_acc[acc]["ssim"] += zf_met["ssim"]
            zf_slice_sums_by_acc[acc]["nmse"] += zf_met["nmse"]
            if self.tracker is not None:
                self.tracker.update(acc, tgt_crop, pred_crop, zf_crop, met, fname)

            # ===== FIGURE EVERY vis_every ITERATIONS =====
            _slice = batch.get("slice", 0)
            if isinstance(_slice, (list, tuple)):
                _slice = _slice[0]
            if torch.is_tensor(_slice):
                sidx = int(_slice.flatten()[0].item())
            else:
                sidx = int(_slice)
            if (it % self.cfg.vis_every) == 0:
                vis.save_figure(
                    recon=pred_crop[0],
                    masked=zf_crop[0],
                    gt=tgt_crop[0],
                    metrics=met,
                    sidx=sidx,
                    filename=f"train_e{epoch:03d}_it{it:05d}_s{self.global_step:07d}_sl{sidx:04d}_{fname[:20]}",
                    title=f"Train Step {self.global_step}",
                    acceleration=acc,
                )

            if it % max(1, self.cfg.print_every) == 0:
                print(
                    f"  [{split}] e{epoch} it{it}/{len(loader)} acc={acc} "
                    f"ZF_PSNR={zf_met['psnr']:.2f} -> PSNR={met['psnr']:.2f} SSIM={met['ssim']:.4f}"
                )

        n_volumes = int(len(vol_names))

        if average == "volume":
            vol_psnr = [float(np.mean([m['psnr'] for m in mets])) for mets in vol_mets.values() if len(mets) > 0]
            vol_ssim = [float(np.mean([m['ssim'] for m in mets])) for mets in vol_mets.values() if len(mets) > 0]
            vol_nmse = [float(np.mean([m['nmse'] for m in mets])) for mets in vol_mets.values() if len(mets) > 0]

            zf_vol_psnr = [float(np.mean([m['psnr'] for m in mets])) for mets in vol_zf_mets.values() if len(mets) > 0]
            zf_vol_ssim = [float(np.mean([m['ssim'] for m in mets])) for mets in vol_zf_mets.values() if len(mets) > 0]
            zf_vol_nmse = [float(np.mean([m['nmse'] for m in mets])) for mets in vol_zf_mets.values() if len(mets) > 0]

            res = {
                "psnr": float(np.mean(vol_psnr)) if vol_psnr else 0.0,
                "ssim": float(np.mean(vol_ssim)) if vol_ssim else 0.0,
                "nmse": float(np.mean(vol_nmse)) if vol_nmse else 0.0,
                "psnr_std": float(np.std(vol_psnr)) if vol_psnr else 0.0,
                "ssim_std": float(np.std(vol_ssim)) if vol_ssim else 0.0,
                "zf_psnr": float(np.mean(zf_vol_psnr)) if zf_vol_psnr else 0.0,
                "zf_ssim": float(np.mean(zf_vol_ssim)) if zf_vol_ssim else 0.0,
                "zf_nmse": float(np.mean(zf_vol_nmse)) if zf_vol_nmse else 0.0,
                "n_samples": int(len(vol_psnr)),
            }
        else:
            res = {k: v / max(n_slices, 1) for k, v in sums.items()}
            res["psnr_std"] = float(np.std(all_psnr)) if all_psnr else 0.0
            res["ssim_std"] = float(np.std(all_ssim)) if all_ssim else 0.0
            res["zf_psnr"] = zf_sums["psnr"] / max(n_slices, 1)
            res["zf_ssim"] = zf_sums["ssim"] / max(n_slices, 1)
            res["zf_nmse"] = zf_sums["nmse"] / max(n_slices, 1)
            res["n_samples"] = int(n_slices)

        # By-acceleration summary (helps when evaluating 4× and 8× in the same run)
        by_acc: Dict[str, Dict[str, float]] = {}
        acc_list = sorted(set(list(n_slices_by_acc.keys()) + list(vol_mets_by_acc.keys())))
        for a in acc_list:
            a_str = str(int(a))
            if average == "volume":
                vols = vol_mets_by_acc.get(a, {})
                zf_vols = vol_zf_mets_by_acc.get(a, {})
                vol_psnr = [float(np.mean([m['psnr'] for m in mets])) for mets in vols.values() if len(mets) > 0]
                vol_ssim = [float(np.mean([m['ssim'] for m in mets])) for mets in vols.values() if len(mets) > 0]
                vol_nmse = [float(np.mean([m['nmse'] for m in mets])) for mets in vols.values() if len(mets) > 0]

                zf_vol_psnr = [float(np.mean([m['psnr'] for m in mets])) for mets in zf_vols.values() if len(mets) > 0]
                zf_vol_ssim = [float(np.mean([m['ssim'] for m in mets])) for mets in zf_vols.values() if len(mets) > 0]
                zf_vol_nmse = [float(np.mean([m['nmse'] for m in mets])) for mets in zf_vols.values() if len(mets) > 0]

                by_acc[a_str] = {
                    "psnr": float(np.mean(vol_psnr)) if vol_psnr else 0.0,
                    "ssim": float(np.mean(vol_ssim)) if vol_ssim else 0.0,
                    "nmse": float(np.mean(vol_nmse)) if vol_nmse else 0.0,
                    "psnr_std": float(np.std(vol_psnr)) if vol_psnr else 0.0,
                    "ssim_std": float(np.std(vol_ssim)) if vol_ssim else 0.0,
                    "zf_psnr": float(np.mean(zf_vol_psnr)) if zf_vol_psnr else 0.0,
                    "zf_ssim": float(np.mean(zf_vol_ssim)) if zf_vol_ssim else 0.0,
                    "zf_nmse": float(np.mean(zf_vol_nmse)) if zf_vol_nmse else 0.0,
                    "n_samples": int(len(vol_psnr)),
                }
            else:
                n_a = int(n_slices_by_acc.get(a, 0))
                if n_a <= 0:
                    by_acc[a_str] = {
                        "psnr": 0.0, "ssim": 0.0, "nmse": 0.0,
                        "psnr_std": 0.0, "ssim_std": 0.0,
                        "zf_psnr": 0.0, "zf_ssim": 0.0, "zf_nmse": 0.0,
                        "n_samples": 0,
                    }
                else:
                    by_acc[a_str] = {
                        "psnr": float(slice_sums_by_acc[a]["psnr"] / n_a),
                        "ssim": float(slice_sums_by_acc[a]["ssim"] / n_a),
                        "nmse": float(slice_sums_by_acc[a]["nmse"] / n_a),
                        "psnr_std": float(np.std(all_psnr_by_acc[a])) if all_psnr_by_acc[a] else 0.0,
                        "ssim_std": float(np.std(all_ssim_by_acc[a])) if all_ssim_by_acc[a] else 0.0,
                        "zf_psnr": float(zf_slice_sums_by_acc[a]["psnr"] / n_a),
                        "zf_ssim": float(zf_slice_sums_by_acc[a]["ssim"] / n_a),
                        "zf_nmse": float(zf_slice_sums_by_acc[a]["nmse"] / n_a),
                        "n_samples": int(n_a),
                    }

        res["by_acc"] = by_acc

        res["n_slices"] = int(n_slices)
        res["n_volumes"] = int(n_volumes)
        res["average"] = average

        if self.writer:
            self.writer.add_scalar(f"{split}/psnr", res.get("psnr", 0.0), epoch)
            self.writer.add_scalar(f"{split}/ssim", res.get("ssim", 0.0), epoch)
            self.writer.add_scalar(f"{split}/nmse", res.get("nmse", 0.0), epoch)
            self.writer.add_scalar(f"{split}/zf_psnr", res.get("zf_psnr", 0.0), epoch)
            # Per-acc scalars (only if present)
            by_acc = res.get("by_acc", {})
            if isinstance(by_acc, dict):
                for a_str, d in by_acc.items():
                    if not isinstance(d, dict):
                        continue
                    self.writer.add_scalar(f"{split}/psnr_acc{a_str}", d.get("psnr", 0.0), epoch)
                    self.writer.add_scalar(f"{split}/ssim_acc{a_str}", d.get("ssim", 0.0), epoch)
                    self.writer.add_scalar(f"{split}/nmse_acc{a_str}", d.get("nmse", 0.0), epoch)

        return res

    def save_results(self, results: Dict[str, Any], filename: str = "final_results"):
        results_dir = self.out / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        json_path = results_dir / f"{filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        csv_path = results_dir / f"{filename}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("split,psnr_mean,psnr_std,ssim_mean,ssim_std,nmse_mean,zf_psnr,zf_ssim,n_samples\n")
            for split, stats in results.items():
                if not isinstance(stats, dict):
                    continue
                f.write(f"{split},{stats.get('psnr', 0):.4f},{stats.get('psnr_std', 0):.4f},"
                        f"{stats.get('ssim', 0):.4f},{stats.get('ssim_std', 0):.4f},"
                        f"{stats.get('nmse', 0):.4f},{stats.get('zf_psnr', 0):.4f},"
                        f"{stats.get('zf_ssim', 0):.4f},{stats.get('n_samples', 0)}\n")

        print(f"[RESULTS] Saved: {json_path} | {csv_path}")

    def fit(self) -> Dict[str, Any]:
        # Build datasets
        train_ds = FastMRIDataset(self.cfg.train_path, list(self.cfg.accelerations), self.cfg.cf_map, seed=self.cfg.seed, max_files=self.cfg.max_files) if self.cfg.train_path else None
        val_ds = FastMRIDataset(self.cfg.val_path, list(self.cfg.accelerations), self.cfg.cf_map, seed=self.cfg.seed + 1, max_files=self.cfg.max_files) if self.cfg.val_path else None
        test_ds = FastMRIDataset(self.cfg.test_path, list(self.cfg.accelerations), self.cfg.cf_map, seed=self.cfg.seed + 2, max_files=self.cfg.max_files) if self.cfg.test_path else None

        # Loaders
        if train_ds is not None:
            _train_sampler = CoilGroupedBatchSampler(
                train_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                seed=self.cfg.seed,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=_train_sampler,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                collate_fn=safe_collate,
            )
        else:
            train_loader = None
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=safe_collate) if val_ds else None
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=safe_collate) if test_ds else None

        if train_loader is None and val_loader is None and test_loader is None:
            raise ValueError("Provide at least one of train_path/val_path/test_path")

        print("=" * 70)
        print("DDC-KSE-ViT-GAN: TRAIN / VALIDATE / TEST (Integrated)")
        print(f"Output: {self.out}")
        print("=" * 70)

        all_results: Dict[str, Any] = {}

        for ep in range(self.start_epoch, self.cfg.epochs):
            e = ep + 1
            print(f"\nEpoch {e}/{self.cfg.epochs}")
            print("-" * 50)

            if train_loader:
                tr = self.train_epoch(train_loader, e)
                all_results[f"train_e{e}"] = tr
                print(f"[EPOCH {e}] Train: PSNR={tr.get('psnr',0):.2f} SSIM={tr.get('ssim',0):.4f}")

            if val_loader:
                va = self.validate(val_loader, e, "val")
                all_results[f"val_e{e}"] = va
                print(f"[EPOCH {e}] Val: PSNR={va.get('psnr',0):.2f}±{va.get('psnr_std',0):.2f} SSIM={va.get('ssim',0):.4f}±{va.get('ssim_std',0):.4f} | ZF_PSNR={va.get('zf_psnr',0):.2f}")
                by_acc = va.get("by_acc", {}) if isinstance(va, dict) else {}
                if isinstance(by_acc, dict) and len(by_acc) > 0:
                    for a_str in sorted(by_acc.keys(), key=lambda x: int(x)):
                        d = by_acc.get(a_str, {})
                        if not isinstance(d, dict):
                            continue
                        print(f"         [Val {a_str}x] PSNR={d.get('psnr',0):.2f}±{d.get('psnr_std',0):.2f} SSIM={d.get('ssim',0):.4f}±{d.get('ssim_std',0):.4f} NMSE={d.get('nmse',0):.5f} | ZF_PSNR={d.get('zf_psnr',0):.2f}")

                self.ckpt_mgr.save(
                    self.generator, self.discriminator,
                    self.optim_g, self.optim_d,
                    self.scaler_g, self.scaler_d,
                    epoch=e, step=self.global_step,
                    metrics={"psnr": va["psnr"], "ssim": va["ssim"], "nmse": va["nmse"]},
                    extra={"val": va},
                )

        # Test
        if test_loader:
            print("\n" + "=" * 70)
            print("FINAL TEST EVALUATION")
            print("=" * 70)
            te = self.validate(test_loader, self.cfg.epochs, "test")
            all_results["test"] = te
            print(f"[TEST] PSNR={te.get('psnr',0):.2f}±{te.get('psnr_std',0):.2f} SSIM={te.get('ssim',0):.4f}±{te.get('ssim_std',0):.4f} | ZF_PSNR={te.get('zf_psnr',0):.2f}")
            by_acc = te.get("by_acc", {}) if isinstance(te, dict) else {}
            if isinstance(by_acc, dict) and len(by_acc) > 0:
                for a_str in sorted(by_acc.keys(), key=lambda x: int(x)):
                    d = by_acc.get(a_str, {})
                    if not isinstance(d, dict):
                        continue
                    print(f"       [Test {a_str}x] PSNR={d.get('psnr',0):.2f}±{d.get('psnr_std',0):.2f} SSIM={d.get('ssim',0):.4f}±{d.get('ssim_std',0):.4f} NMSE={d.get('nmse',0):.5f} | ZF_PSNR={d.get('zf_psnr',0):.2f}")

        self.save_results(all_results)
        if self.writer:
            try:
                self.writer.close()
            except Exception:
                pass
        # ── Close iteration logger ───────────────────────────────────
        try:
            self.iter_logger.close()
        except Exception:
            pass

        print("=" * 70)
        print(f"Training complete. Best monitored {self.ckpt_mgr.monitor}: {self.ckpt_mgr.best:.4f}")
        print("=" * 70)
        return all_results


# ============================================================
# MAIN (paths are pre-filled; override with CLI)
# ============================================================

def _parse_cf_map(pairs: List[str]) -> Dict[int, float]:
    """
    Parse --cf_map like: 4:0.08 8:0.04
    """
    out = {}
    for s in pairs:
        if ":" not in s:
            continue
        a, c = s.split(":", 1)
        out[int(a)] = float(c)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DDC-KSE-ViT-GAN: Train/Val/Test (Single-file integrated)")
    # Paths
    parser.add_argument("--train_path", type=str, default=os.environ.get("FASTMRI_TRAIN", r"D:\train\multicoil_train"))
    parser.add_argument("--val_path", type=str, default=os.environ.get("FASTMRI_VAL", r"C:\Users\Muham\PycharmProjects\fastmri_f\val\multicoil_val"))
    parser.add_argument("--test_path", type=str, default=os.environ.get("FASTMRI_TEST", r"C:\Users\Muham\PycharmProjects\fastmri_f\test\multicoil_test_full"))
    parser.add_argument("--out_dir", type=str, default="./outputs_vit_gan_integrated_cascade4")

    # Data
    parser.add_argument("--batch_size", type=int, default=2, choices=[1, 2, 4, 8],
                        help="Training batch size. 2 is safe on RTX 3090. Try 4 with --no_amp disabled.")
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("--accelerations", type=int, nargs="+", default=[8])
    parser.add_argument("--cf_map", type=str, nargs="*", default=["8:0.04"])
    parser.add_argument("--eval_crop", type=int, nargs=2, default=[320, 320])
    parser.add_argument(
        "--train_crop", type=int, nargs=2, default=[320, 320],
        help="Crop (H W) used during training loss/metrics/vis. Defaults to eval_crop."
    )
    parser.add_argument(
        "--eval_average", type=str, default="volume", choices=["slice", "volume"],
        help="Metric aggregation in validation/test: per-slice or per-volume mean."
    )
    parser.add_argument("--max_files", type=int, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_g", type=float, default=2e-3)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--empty_cache_every", type=int, default=5, help="Call torch.cuda.empty_cache() every N steps (0 disables).")
    parser.add_argument("--no_skip_oom", action="store_true", help="If set, do NOT skip CUDA OOM batches; crash instead.")

    # GAN
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--no_gan", action="store_true")
    parser.add_argument("--gan_start_epoch", type=int, default=25)
    parser.add_argument("--d_steps_per_g", type=int, default=1)

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--adversarial_weight", type=float, default=0.01)

    # AMP
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])

    # Visualization + ckpt
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--max_vis", type=int, default=100000000)
    parser.add_argument("--time_ckpt_minutes", type=int, default=25)
    parser.add_argument("--ckpt_keep", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_best", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = TrainConfigAllInOne(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerations=tuple(args.accelerations),
        cf_map=_parse_cf_map(args.cf_map),
        eval_crop_hw=(int(args.eval_crop[0]), int(args.eval_crop[1])),
        train_crop_hw=(int(args.train_crop[0]), int(args.train_crop[1])) if args.train_crop is not None else (int(args.eval_crop[0]), int(args.eval_crop[1])),
        eval_average=str(args.eval_average),
        max_files=args.max_files,

        epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        grad_clip=args.grad_clip,

        empty_cache_every=args.empty_cache_every,
        skip_oom=not args.no_skip_oom,

        use_gan_refinement=(args.use_gan and not args.no_gan) if (args.use_gan or args.no_gan) else True,
        gan_start_epoch=args.gan_start_epoch,
        d_steps_per_g=args.d_steps_per_g,

        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        adversarial_weight=args.adversarial_weight,

        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,

        vis_every=args.vis_every,
        max_vis_per_epoch=args.max_vis,
        time_ckpt_minutes=args.time_ckpt_minutes,
        ckpt_keep=args.ckpt_keep,
        resume=args.resume,
        resume_best=args.resume_best,
        seed=args.seed,
    )

    # Auto AMP tuning for Ampere+
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            cfg.use_amp = True
            cfg.amp_dtype = "bf16"

    trainer = TrainerViTGANIntegrated(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()