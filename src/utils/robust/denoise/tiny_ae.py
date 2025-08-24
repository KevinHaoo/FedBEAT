# -*- coding: utf-8 -*-
# src/utils/robust/denoise/tiny_ae.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

class TinyAE(nn.Module):
    """
    极简轻量自编码器（卷积版），兼容 MNIST(1x28x28) / CIFAR10(3x32x32)。
    - 黑盒评测场景：作为“输入净化器”，仅前向推理，不参与反传到主干。
    - 结构：ConvDown x2 -> Conv -> DeconvUp x2 -> Sigmoid
    - 不使用 FC latent，避免对尺寸敏感；计算量极低。

    Args:
        in_ch: 输入通道（MNIST=1, CIFAR10=3）
        base:  主通道宽度
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, stride=2, padding=1, bias=False),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base*2, kernel_size=3, stride=2, padding=1, bias=False), # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, kernel_size=4, stride=2, padding=1, bias=False), # H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_ch, kernel_size=4, stride=2, padding=1, bias=False),  # H
            nn.Sigmoid(),  # 输出到 [0,1]
        )

        # 小心初始化：Kaiming for ReLU, unit-normal for deconv is OK
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def encode(self, x: Tensor) -> Tensor:
        return self.enc(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


if __name__ == "__main__":
    # --- minimal self-test (≤10 lines) ---
    torch.manual_seed(0)
    x1 = torch.rand(2, 1, 28, 28)
    x3 = torch.rand(2, 3, 32, 32)
    ae1 = TinyAE(in_ch=1, base=16)
    ae3 = TinyAE(in_ch=3, base=16)
    y1, y3 = ae1(x1), ae3(x3)
    print(y1.shape, y3.shape, torch.isfinite(y1).all().item() and torch.isfinite(y3).all().item())
