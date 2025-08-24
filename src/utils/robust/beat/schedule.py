# -*- coding: utf-8 -*-
# file: src/robust/beat/schedule.py
from __future__ import annotations
from typing import Dict, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.robust.dbfat.losses import dbfat_blackbox_loss
from src.utils.robust.wrappers import AttackModelWrapper

from contextlib import nullcontext
try:
    from torch.amp import autocast, GradScaler
except Exception:
    autocast = nullcontext  # 兼容 CPU
    class GradScaler:       # 兼容 CPU
        def __init__(self, enabled=False): self.enabled = False
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass

@torch.no_grad()
def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    """确保 logits 形状为 [B, C]。"""
    if logits.dim() != 2:
        logits = logits.view(logits.size(0), -1)
    return logits


def local_train_two_stage(
    backbone: nn.Module,
    beat: Optional[nn.Module],
    train_loader,
    opt_backbone: torch.optim.Optimizer,
    opt_beat: Optional[torch.optim.Optimizer],
    attack_fn: Callable[[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor, float, int], torch.Tensor],
    eps: float,
    steps: int,
    device: torch.device,
    beta: float,
    epochs_fedavg: int,
    epochs_beat: int,
) -> Dict[str, float]:
    """
    两阶段本地训练：
      A) FedAvg 阶段：训练 backbone（干净样本）；
      B) BEAT 阶段：冻结 backbone；用黑盒攻击生成 x_adv，计算 DBFAT 黑盒损失，仅更新 beat。
    仅返回指标；上层只上传 backbone 参数。
    """
    metrics = {
        "loss_fedavg": 0.0,
        "loss_beat_total": 0.0,
        "loss_beat_ce": 0.0,
        "loss_beat_kl": 0.0,
        "rho_mean": 0.0,
        "num_batches_fedavg": 0.0,
        "num_batches_beat": 0.0,
    }
    backbone.to(device)
    if beat is not None:
        beat.to(device)

    # -------------------------
    # A) FedAvg 阶段：训练 backbone（干净样本）
    # -------------------------
    criterion = nn.CrossEntropyLoss(reduction="mean")
    backbone.train()
    for ep in range(max(0, int(epochs_fedavg))):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt_backbone.zero_grad(set_to_none=True)
            logits = backbone(x)
            logits = logits if logits.dim() == 2 else logits.view(logits.size(0), -1)
            loss = criterion(logits, y)
            loss.backward()
            opt_backbone.step()

            metrics["loss_fedavg"] += float(loss.detach().item())
            metrics["num_batches_fedavg"] += 1.0

    # -------------------------
    # B) BEAT 阶段：冻结 backbone；仅更新 beat（黑盒）
    # -------------------------
    # if beat is not None and opt_beat is not None and epochs_beat > 0:
    #     # 冻结 backbone；攻击与 logits_main 获取均使用 AttackModelWrapper（内部 no_grad + eval）
    #     for p in backbone.parameters():
    #         p.requires_grad_(False)
    #     backbone.eval()
    #     beat.train()
    #
    #     atk_model = AttackModelWrapper(backbone).to(device)
    #
    #     for ep in range(int(epochs_beat)):
    #         for x, y in train_loader:
    #             x = x.to(device, non_blocking=True)
    #             y = y.to(device, non_blocking=True)
    #
    #             # ---- 黑盒攻击：只暴露 model_fn(x)->logits ----
    #             def model_fn(inp: torch.Tensor) -> torch.Tensor:
    #                 return atk_model(inp)  # AttackModelWrapper 已 no_grad + eval
    #
    #             with torch.no_grad():
    #                 # 攻击器内部会处理 clamp/投影等；此处只需传 eps/steps
    #                 x_adv = attack_fn(model_fn, x, y, float(eps), int(steps))
    #                 # 主干 logits（无梯度）
    #                 logits_main = atk_model(x_adv)
    #                 logits_main = _ensure_2d_logits(logits_main)
    #
    #             # ---- BEAT 残差 + DBFAT 黑盒损失（只更新 beat）----
    #             logits_aug = logits_main + beat(logits_main)  # beat 输出残差
    #             losses = dbfat_blackbox_loss(
    #                 logits_main=logits_main,  # 内部会 detach 双保险
    #                 logits_aug=logits_aug,
    #                 y=y,
    #                 beta=float(beta),
    #             )
    #
    #             opt_beat.zero_grad(set_to_none=True)
    #             losses["loss_total"].backward()
    #             # 可选：梯度裁剪，避免数值爆炸
    #             torch.nn.utils.clip_grad_norm_(beat.parameters(), max_norm=5.0)
    #             opt_beat.step()
    #
    #             metrics["loss_beat_total"] += float(losses["loss_total"].detach().item())
    #             metrics["loss_beat_ce"] += float(losses["loss_ce"].detach().item())
    #             metrics["loss_beat_kl"] += float(losses["loss_kl"].detach().item())
    #             metrics["rho_mean"] += float(losses["rho_mean"])
    #             metrics["num_batches_beat"] += 1.0
    #
    #     # 训练结束，恢复 requires_grad（以便下一轮 FedAvg 继续训练）
    #     for p in backbone.parameters():
    #         p.requires_grad_(True)

    if epochs_beat > 0 and beat is not None and opt_beat is not None:
        # === NEW: 读取提速选项（由 client 在调用前挂到 beat 上） ===
        _speed = getattr(beat, "_speed_opts", {})  # dict: {'beat_stride', 'beat_client_prob', 'amp'}
        beat_stride = int(_speed.get("beat_stride", 1))
        beat_client_prob = float(_speed.get("beat_client_prob", 1.0))
        use_amp = bool(_speed.get("amp", False)) and torch.cuda.is_available()
        scaler = GradScaler('cuda', enabled=use_amp)
        # 若本轮客户端按概率跳过 BEAT，直接进入收尾
        if torch.rand(1).item() > beat_client_prob:
            # --- 收尾平均（保持原逻辑） ---
            if metrics["num_batches_fedavg"] > 0:
                metrics["loss_fedavg"] /= metrics["num_batches_fedavg"]
            return metrics
        # === /NEW ===

        # 记录进入前的训练态与 requires_grad
        was_training = backbone.training
        prev_requires = [p.requires_grad for p in backbone.parameters()]

        # 冻结 backbone，只训 beat
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        beat.train()

        # try:
        #     for _ in range(epochs_beat):
        #         for x, y in train_loader:
        #             x = x.to(device, non_blocking=True)
        #             y = y.to(device, non_blocking=True)
        #
        #             # 用黑盒攻击产生对抗样本
        #             x_adv = attack_fn(lambda t: backbone(t).detach(), x, y, eps=eps, steps=steps)
        #
        #             # 前向：main logits（不反传） + beat 残差
        #             with torch.no_grad():
        #                 logits_main = backbone(x_adv)
        #
        #             residual = beat(logits_main)  # 只对 beat 求梯度
        #             logits_aug = logits_main + residual
        #
        #             # DBFAT 黑盒损失
        #             loss_pack = dbfat_blackbox_loss(
        #                 logits_main=logits_main.detach(),  # 再次确保不反传到 backbone
        #                 logits_aug=logits_aug,
        #                 y=y,
        #                 beta=beta,
        #             )
        #
        #             opt_beat.zero_grad(set_to_none=True)
        #             loss_pack["loss_total"].backward()
        #             opt_beat.step()
        #
        #             # 你原来记录 stats 的代码保持
        #             metrics["loss_beat_total"] += float(loss_pack["loss_total"].detach().item())
        #             metrics["loss_beat_ce"] += float(loss_pack["loss_ce"].detach().item())
        #             metrics["loss_beat_kl"] += float(loss_pack["loss_kl"].detach().item())
        #             metrics["rho_mean"] += float(loss_pack["rho_mean"])
        #             metrics["num_batches_beat"] += 1.0
        # finally:
        #     # 无论是否异常，务必恢复训练态与 requires_grad
        #     for p, req in zip(backbone.parameters(), prev_requires):
        #         p.requires_grad = req
        #     backbone.train(was_training)
        try:
            for _ in range(int(epochs_beat)):
                for it, (x, y) in enumerate(train_loader):
                    # === NEW: 每 beat_stride 个 batch 做一次攻击 ===
                    if beat_stride > 1 and (it % beat_stride) != 0:
                        continue
                    # === /NEW ===

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    # 黑盒攻击
                    x_adv = attack_fn(lambda t: backbone(t).detach(), x, y, eps=float(eps), steps=int(steps))

                    with torch.no_grad():
                        logits_main = backbone(x_adv)
                        logits_main = _ensure_2d_logits(logits_main)

                    # 残差 + 损失（AMP 可选）
                    residual = beat(logits_main)
                    logits_aug = logits_main + residual

                    # === NEW(AMP)：只对 BEAT 头反传部分做半精度 ===
                    if use_amp:
                        with autocast('cuda'):
                            pack = dbfat_blackbox_loss(logits_main=logits_main.detach(),
                                                       logits_aug=logits_aug, y=y, beta=float(beta))
                        opt_beat.zero_grad(set_to_none=True)
                        scaler.scale(pack["loss_total"]).backward()
                        torch.nn.utils.clip_grad_norm_(beat.parameters(), 5.0)
                        scaler.step(opt_beat)
                        scaler.update()
                    else:
                        pack = dbfat_blackbox_loss(logits_main=logits_main.detach(),
                                                   logits_aug=logits_aug, y=y, beta=float(beta))
                        opt_beat.zero_grad(set_to_none=True)
                        pack["loss_total"].backward()
                        torch.nn.utils.clip_grad_norm_(beat.parameters(), 5.0)
                        opt_beat.step()
                    # === /NEW ===

                    metrics["loss_beat_total"] += float(pack["loss_total"].detach().item())
                    metrics["loss_beat_ce"] += float(pack["loss_ce"].detach().item())
                    metrics["loss_beat_kl"] += float(pack["loss_kl"].detach().item())
                    metrics["rho_mean"] += float(pack["rho_mean"])
                    metrics["num_batches_beat"] += 1.0
        finally:
            # 无论是否异常，务必恢复训练态与 requires_grad
            for p, req in zip(backbone.parameters(), prev_requires):
                p.requires_grad = req
            backbone.train(was_training)

    # 聚合平均
    if metrics["num_batches_fedavg"] > 0:
        metrics["loss_fedavg"] /= metrics["num_batches_fedavg"]
    if metrics["num_batches_beat"] > 0:
        nb = metrics["num_batches_beat"]
        metrics["loss_beat_total"] /= nb
        metrics["loss_beat_ce"] /= nb
        metrics["loss_beat_kl"] /= nb
        metrics["rho_mean"] /= nb

    return metrics


# ------------------------ ≤10 行最小自检 ------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 造个小数据 & DataLoader
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    # 2) 假主干：28*28 -> 10 类
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        def forward(self, x): return self.net(x)

    # 3) 假 BEAT（残差头）：10 -> 10
    class DummyBeat(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 10))
        def forward(self, z): return self.mlp(z)

    # 4) 假攻击器：恒等（不改动 x）
    def fake_attack(model_fn, x, y, eps, steps):
        _ = model_fn(x)  # 走一遍接口
        return x  # 不做扰动

    backbone = DummyBackbone().to(device)
    beat = DummyBeat().to(device)
    opt_b = torch.optim.SGD(backbone.parameters(), lr=0.1)
    opt_h = torch.optim.SGD(beat.parameters(), lr=0.1)

    stats = local_train_two_stage(
        backbone=backbone,
        beat=beat,
        train_loader=loader,
        opt_backbone=opt_b,
        opt_beat=opt_h,
        attack_fn=fake_attack,
        eps=8/255,
        steps=5,
        device=device,
        beta=1.0,
        epochs_fedavg=1,
        epochs_beat=1,
    )
    print({k: round(v, 4) for k, v in stats.items() if "num_batches" not in k})
