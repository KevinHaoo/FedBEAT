# -*- coding: utf-8 -*-
# src/utils/robust/denoise/median.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor

@torch.no_grad()
def median_filter(x: Tensor, kernel: int = 3) -> Tensor:
    """
    Channel-wise median filtering for images (BCHW).
    Inputs:
        - x: [B,C,H,W] (建议在 [0,1] 像素域)
        - kernel: odd integer (>=1)
    Returns:
        - x_denoised: [B,C,H,W]
    """
    assert x.dim() == 4, "x must be BCHW"
    assert kernel >= 1 and kernel % 2 == 1, "kernel must be odd"
    B, C, H, W = x.shape
    pad = kernel // 2

    # 反射填充（减少边界伪影）
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")  # [B,C,H+2p,W+2p]

    # unfold: 每个滑窗成列
    patches = F.unfold(x_pad, kernel_size=kernel, stride=1) # [B, C*kernel*kernel, H*W]
    # 重新排列: [B*C, K*K, H*W]
    patches = patches.view(B, C, kernel*kernel, H*W).permute(0,1,3,2).reshape(B*C, H*W, kernel*kernel)

    # 中位数
    med = patches.median(dim=-1).values                     # [B*C, H*W]

    # 回到 [B,C,H,W]
    med = med.reshape(B, C, H, W)
    return med.clamp(0.0, 1.0)


if __name__ == "__main__":
    # --- minimal self-test (≤10 lines) ---
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8)
    y = median_filter(x, kernel=3)
    print(y.shape, torch.all((0<=y)&(y<=1)).item())
