# src/utils/robust/beat/module.py
# 新增（非四库）：BEAT-style 后插残差模块（logits -> residual logits）
from __future__ import annotations

import torch
import torch.nn as nn

class BEATModule(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 128, use_dropout: bool = True):
        """
        Args:
            num_classes: 类别数 C（即 logits 维度）
            hidden: 隐层维度；>0 两层 MLP；==0 退化为线性 C->C（仍输出残差）
            use_dropout: 是否在隐层后加入 Dropout(0.2)
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.use_dropout = bool(use_dropout)

        if self.hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(self.num_classes, self.hidden, bias=True),
                nn.GELU(),
                nn.Dropout(p=0.2) if self.use_dropout else nn.Identity(),
                nn.Linear(self.hidden, self.num_classes, bias=True),
            )
        else:
            # 退化：单层线性映射，仍返回“残差 logits”
            self.net = nn.Sequential(
                nn.Linear(self.num_classes, self.num_classes, bias=True)
            )

        init_beat(self)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Input: [B, C] logits; Output: [B, C] residual to add."""
        if logits.dim() != 2 or logits.size(-1) != self.num_classes:
            raise ValueError(
                f"BEATModule expects input [B, {self.num_classes}], got {tuple(logits.shape)}"
            )
        return self.net(logits)


def init_beat(m: BEATModule) -> None:
    """
    初始化策略：
      - 两层 MLP：首层 Xavier 正态（对 GELU 稳定），末层权重与偏置置零，保证初始 residual≈0；
      - 单层线性：直接把该层权重与偏置置零（初始 residual=0），训练可正常更新。
    """
    linear_layers = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    if not linear_layers:
        return

    if len(linear_layers) == 1:
        only = linear_layers[0]
        nn.init.zeros_(only.weight)
        if only.bias is not None:
            nn.init.zeros_(only.bias)
    else:
        first, last = linear_layers[0], linear_layers[-1]
        nn.init.xavier_normal_(first.weight, gain=1.0)
        if first.bias is not None:
            nn.init.zeros_(first.bias)
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


# ------------------------ 最小自检（≤10行） ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 10)  # logits [B=8, C=10]
    beat = BEATModule(num_classes=10, hidden=64, use_dropout=True)
    y = beat(x)
    print(y.shape, bool(torch.isfinite(y).all().item()))  # 期望：torch.Size([8, 10]) True
