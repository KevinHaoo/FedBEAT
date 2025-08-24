# src/robust/dbfat/losses.py
# 新增（非四库）：DBFAT 风格（黑盒改造）损失
# 约定（与项目总规一致）：
#   logits_main = g_θ(x_adv).detach()  # 主干对（对抗样本）的输出，不反传
#   logits_aug  = f_{θ′}(g_θ(x_adv))   # 后插模块对（相同输入）的输出
#
# 定义：
#   ρ(x) = 1 - max_c softmax(logits_main)[c]                   (不排序，直接取max)
#   L_CE = CE( ρ(x) * logits_aug, y )                         (按样本缩放logits后做CE)
#   L_KL = KL( softmax(aug) || softmax(main) )                (batchmean，方向如左)
#   L_total = L_CE + β * L_KL
#
# 数值稳定：
#   - 所有softmax/log_softmax都在dim=1
#   - KL用 torch.nn.functional.kl_div(reduction='batchmean') 的规范用法
#   - logits_main 在此处再次 detach()，防止误传梯度

from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F


def dbfat_blackbox_loss(
    logits_main: torch.Tensor,
    logits_aug: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        logits_main: Tensor [B, C], g_θ(x_adv) 的输出；在此函数内将被 detach()
        logits_aug:  Tensor [B, C], f_{θ′}(g_θ(x_adv)) 的输出（对抗/增强路径）
        y:           Tensor [B],   真实标签（int64）
        beta:        float,        KL 项的权重系数

    Returns:
        dict with keys:
            - 'loss_total': scalar tensor
            - 'loss_ce':    scalar tensor
            - 'loss_kl':    scalar tensor
            - 'rho_mean':   scalar tensor（batch 内 ρ 的均值，便于监控）
    """
    if logits_main.dim() != 2 or logits_aug.dim() != 2:
        raise ValueError(f"logits_main/logits_aug must be 2D [B,C], "
                         f"got {tuple(logits_main.shape)}, {tuple(logits_aug.shape)}")
    if logits_main.shape != logits_aug.shape:
        raise ValueError(f"Shape mismatch: logits_main {tuple(logits_main.shape)} vs logits_aug {tuple(logits_aug.shape)}")
    B, C = logits_main.shape

    if y.dim() != 1 or y.shape[0] != B:
        raise ValueError(f"Label y must be [B], got {tuple(y.shape)} (B={B})")
    if y.dtype != torch.long:
        y = y.long()

    # —— ρ(x) = 1 - max_c softmax(main)[c]（detach，确保不反传到主干）——
    with torch.no_grad():
        p_main = F.softmax(logits_main.detach(), dim=1)  # [B,C]
        rho = 1.0 - p_main.max(dim=1).values             # [B]
        # 夹到 [0,1] 以避免数值越界（理论上已在此区间）
        rho = rho.clamp_(0.0, 1.0)

    # —— L_CE：对 logits_aug 做按样本缩放后再交叉熵 ——
    #   注意：按照你的规范是“ρ * logits_aug”，不是样本权重CE
    logits_scaled = logits_aug * rho.unsqueeze(1)        # [B,C]
    loss_ce = F.cross_entropy(logits_scaled, y, reduction="mean")

    # —— L_KL：KL( softmax(aug) || softmax(main) )（batchmean）——
    #   P = softmax(aug)  -> target
    #   Q = softmax(main) -> input
    #   torch 的 kl_div(log_input, target) 计算 KL(target || input)
    log_q = F.log_softmax(logits_main.detach(), dim=1)   # log Q
    p = F.softmax(logits_aug, dim=1)                     #    P
    loss_kl = F.kl_div(log_q, p, reduction="batchmean")

    loss_total = loss_ce + float(beta) * loss_kl

    return {
        "loss_total": loss_total,
        "loss_ce": loss_ce,
        "loss_kl": loss_kl,
        "rho_mean": rho.mean(),
    }


# ------------------------ 最小自检（≤10行） ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 8, 10
    logits_main = torch.randn(B, C)       # 假装是 g_θ(x_adv) 输出
    logits_aug  = torch.randn(B, C)       # 假装是后插模块输出
    y = torch.randint(0, C, (B,))
    out = dbfat_blackbox_loss(logits_main, logits_aug, y, beta=1.0)
    print({k: float(v) if v.ndim == 0 else v.shape for k, v in out.items()})
    # 预期：各项为标量（rho_mean ~ 0.x），不会报错
