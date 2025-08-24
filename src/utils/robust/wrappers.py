# -*- coding: utf-8 -*-
# file: src/robust/wrappers.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class ForwardWrapper(nn.Module):
    """
    ForwardWrapper(backbone, beat=None)
    - backbone: g_θ，输入 x->[B,C,...] -> 输出 logits_main [B,C]
    - beat:     可选后插模块（如 BEATModule），接收 logits_main 输出 residual_logits [B,C]
                最终 logits = logits_main + residual_logits
    """
    def __init__(self, backbone: nn.Module, beat: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.beat = beat

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅主干 g_θ 的输出（不给 BEAT）。
        注：这里默认 no_grad 以契合“黑盒评估/攻击只前向”的使用场景。
        在需要训练主干的阶段，请直接调用 self.backbone(x) 获得可求梯度的 logits。
        """
        self.backbone.eval()  # 前向推理的默认安全态；训练时请直接用 backbone
        logits_main = self.backbone(x)
        if logits_main.dim() != 2:
            # 常见主干已输出 [B,C]；若出现 [B,C, ...]，显式拉平剩余维度
            logits_main = logits_main.view(logits_main.size(0), -1)
        return logits_main

    def forward(self, x: torch.Tensor, use_beat: bool = True) -> torch.Tensor:
        """
        标准前向：
          - use_beat=True  : 返回 logits_main + beat(residual)
          - use_beat=False : 只返回 logits_main
        """
        logits_main = self.backbone(x)
        if logits_main.dim() != 2:
            logits_main = logits_main.view(logits_main.size(0), -1)

        if use_beat and (self.beat is not None):
            # BEAT 只吃 [B,C]，输出残差，同形状
            residual = self.beat(logits_main)
            if residual.shape != logits_main.shape:
                raise ValueError(
                    f"BEAT residual shape {tuple(residual.shape)} "
                    f"!= logits shape {tuple(logits_main.shape)}"
                )
            return logits_main + residual
        else:
            return logits_main


class AttackModelWrapper(nn.Module):
    """
    AttackModelWrapper(backbone)
    - 给黑盒攻击器用：只暴露 g_θ 的 logits 前向，不包含 BEAT/其它防御分支。
    - 内部强制 no_grad，避免任何梯度泄露。
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone.eval()
            logits = self.backbone(x)
            if logits.dim() != 2:
                logits = logits.view(logits.size(0), -1)
            return logits


# ------------------------ ≤10 行最小自检 ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 假主干：28*28 -> 10 类
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        def forward(self, x): return self.net(x)

    # 假 BEAT：恒等残差
    class DummyBeat(nn.Module):
        def forward(self, logits): return torch.zeros_like(logits)

    x = torch.randn(8, 1, 28, 28)
    g = DummyBackbone()
    beat = DummyBeat()

    fw = ForwardWrapper(g, beat)
    am = AttackModelWrapper(g)

    out_main = fw.forward_logits(x)
    out_with = fw(x, use_beat=True)
    out_wo = fw(x, use_beat=False)
    out_atk = am(x)

    print(tuple(out_main.shape), tuple(out_with.shape), tuple(out_wo.shape), tuple(out_atk.shape))
