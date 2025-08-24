# -*- coding: utf-8 -*-
# src/utils/robust/rs/smoothing.py
from __future__ import annotations
import torch
from torch import Tensor

@torch.no_grad()
def randomized_smoothing(model_fn, x: Tensor, sigma: float, n: int) -> Tensor:
    """
    Test-time randomized smoothing (black-box friendly).
    Inputs:
        - model_fn: callable(Tensor[B,C,H,W in [0,1]]) -> logits [B, num_classes]
                    仅前向，无梯度（本函数已 no_grad）
        - x:  [B,C,H,W] in [0,1]  —— 若是标准化张量，请先反标准化到 [0,1]
        - sigma: 高斯噪声的标准差（像素域）
        - n: 采样次数（越大越稳）
    Returns:
        - logits_smooth: [B, num_classes]，通过对概率平均后取 log 得到
                         （argmax 与平均概率一致，常用于评测）
    """
    assert x.dim() == 4, "x must be BCHW"
    assert n >= 1 and sigma >= 0.0

    device = x.device
    B = x.size(0)

    # 概率累加：E_{η~N(0,σ^2)}[softmax(f(x+η))]
    probs_acc = None
    for _ in range(n):
        noise = torch.randn_like(x, device=device) * sigma
        x_noisy = (x + noise).clamp(0.0, 1.0)
        logits = model_fn(x_noisy)                         # [B, C]
        probs  = torch.softmax(logits, dim=1)              # [B, C]
        probs_acc = probs if probs_acc is None else (probs_acc + probs)

    probs_mean = probs_acc / float(n)                      # [B, C]
    # 返回 log(prob) 作为“稳定的 logits”（仅影响标量偏置，不影响 argmax）
    logits_smooth = torch.log(probs_mean.clamp_min(1e-12))
    return logits_smooth


if __name__ == "__main__":
    # --- minimal self-test (≤10 lines) ---
    torch.manual_seed(0)
    B,C,H,W = 4,1,28,28
    x01 = torch.rand(B,C,H,W)
    def dummy_fn(z):  # 伪黑盒前向
        return torch.randn(B, 10)
    out = randomized_smoothing(dummy_fn, x01, sigma=0.25, n=8)
    print(out.shape, torch.isfinite(out).all().item())
