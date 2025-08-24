# -*- coding: utf-8 -*-
# file: src/client/fedbeat.py
from __future__ import annotations
import argparse
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.client.fedavg import FedAvgClient

# 我们在 Prompt 1/3/5 已实现的模块
from src.utils.robust.beat.module import BEATModule
from src.utils.robust.beat.schedule import local_train_two_stage
from src.utils.robust.attacks.blackboxbench.wrappers import (
    run_attack_square, run_attack_spsa, run_attack_onepixel, run_attack_pixle
)

logger = logging.getLogger(__name__)


class FedBEATClient(FedAvgClient):
    """
    两阶段本地训练客户端：
      A) 先按 FedAvgClient.fit() 进行常规本地训练（上传仅 backbone）
      B) 再在本地执行 BEAT 阶段：冻结 backbone，黑盒攻击 + DBFAT，只更新 BEAT 残差头
         —— BEAT 参数仅本地保存，不上传
    """

    @staticmethod
    def get_hyperparams(argv=None):
        # 不再暴露旧键；全部从 self.args 读取新键（fedbeat.attacks.*, fedbeat.epochs_beat, fedbeat.loss.kl.beta）
        if argv is None:
            argv = []
        parser = argparse.ArgumentParser(add_help=False)
        ns, _ = parser.parse_known_args(argv)
        return ns

    # ----------------------------------------------------------------------
    # 覆写 fit：先父类 FedAvg，再 BEAT 阶段（本地）
    # ----------------------------------------------------------------------
    def fit(self):
        # 1) FedAvg 阶段（保持与基类完全一致的行为、日志、上传协议）
        super().fit()

        # 2) 本地执行 BEAT 阶段（不影响上传内容）
        try:
            self._run_beat_stage()
        except Exception as e:
            logger.exception(f"[FedBEAT-Client {getattr(self, 'client_id', '?')}] BEAT stage failed: {e}")

        # 3) 保险：清理 BN 的 buffers（避免后续前向时报 'leaf Variable'）
        self._sanitize_batchnorm_buffers(self.model)

    # ----------------------------------------------------------------------
    # 私有：执行 BEAT 阶段
    # ----------------------------------------------------------------------
    # 来自性能优化（3.1 / 3.2 / 3.5）：以最小开销在客户端侧降低 BEAT 触发频率与成本
    def _run_beat_stage(self) -> None:
        cfg = self._read_fb_cfg()
        if cfg["epochs_beat"] <= 0:
            return  # 本轮不启用 BEAT

        # === NEW: 客户端概率门控（全局期望减少一半/三分之一客户端跑 BEAT） ===
        p = float(getattr(self.args.fedbeat, "speed", {}).get("beat_client_prob", 1.0)) \
            if hasattr(self.args, "fedbeat") else 1.0
        if torch.rand(1).item() > p:
            return
        # === /NEW ===

        # 惰性初始化 BEAT 残差头 + 专属优化器（仅本地，不上传）
        self._lazy_init_beat_head()

        # === NEW: 把提速选项挂在 beat 头上，供 schedule 读取 ===
        speed = getattr(self.args.fedbeat, "speed", {}) if hasattr(self.args, "fedbeat") else {}
        setattr(self._beat_head, "_speed_opts", {
            "beat_stride": int(speed.get("beat_stride", 1)),
            "beat_client_prob": float(speed.get("beat_client_prob", 1.0)),
            "amp": bool(speed.get("amp", False)),
        })
        # === /NEW ===

        # 选择攻击器
        attack_dispatch = {
            "square":   run_attack_square,
            "spsa":     run_attack_spsa,
            "onepixel": run_attack_onepixel,
            "pixle":    run_attack_pixle,
        }
        atk_name = cfg["attack_name"]
        if atk_name not in attack_dispatch:
            raise ValueError(f"[FedBEAT-Client] Unknown attack '{atk_name}'.")
        attack_fn = attack_dispatch[atk_name]

        # 只训练 BEAT 头（epochs_fedavg=0；冻结骨干由 schedule 内部处理并自动恢复）
        stats = local_train_two_stage(
            backbone=self.model,
            beat=self._beat_head,
            train_loader=self.trainloader,
            opt_backbone=self.optimizer,           # 不会被用到（epochs_fedavg=0）
            opt_beat=self._beat_optim,
            attack_fn=attack_fn,
            eps=cfg["eps"],
            steps=cfg["steps"],
            device=self.device,
            beta=cfg["beta"],
            epochs_fedavg=0,
            epochs_beat=cfg["epochs_beat"],
        )

        logger.info(
            "[FedBEAT-Client %s] BEAT stats: total=%.4f, ce=%.4f, kl=%.4f, rho=%.4f",
            getattr(self, "client_id", "?"),
            float(stats.get("loss_beat_total", 0.0)),
            float(stats.get("loss_beat_ce", 0.0)),
            float(stats.get("loss_beat_kl", 0.0)),
            float(stats.get("rho_mean", 0.0)),
        )

    # ----------------------------------------------------------------------
    # 配置读取（新键优先，旧键回退；与 server/fedbeat.py 对齐）
    # ----------------------------------------------------------------------
    def _read_fb_cfg(self) -> Dict[str, Any]:
        fb = getattr(self.args, "fedbeat", {}) if hasattr(self, "args") else {}
        atk = fb.get("attacks", {}) if isinstance(fb, dict) else {}

        attack_name = atk.get("active", fb.get("attack", "square"))
        eps = float(atk.get("eps", fb.get("eps", 8 / 255)))
        steps = int(atk.get("steps", fb.get("steps", 1000)))
        epochs_beat = int(fb.get("epochs_beat", fb.get("epochs_fedavg", 0)))
        kl_beta = float(fb.get("loss", {}).get("kl", {}).get("beta", fb.get("beta", 1.5)))

        return {
            "attack_name": attack_name,
            "eps": eps,
            "steps": steps,
            "epochs_beat": epochs_beat,
            "beta": kl_beta,
        }

    # ----------------------------------------------------------------------
    # 惰性初始化 BEAT 头及其优化器（仅本地）
    # ----------------------------------------------------------------------
    def _lazy_init_beat_head(self):
        if hasattr(self, "_beat_head") and self._beat_head is not None:
            # 已初始化；遵循 reset_optimizer_on_global_epoch：仅重置优化器状态
            if getattr(self.args.common, "reset_optimizer_on_global_epoch", True) and hasattr(self, "_beat_optim"):
                self._beat_optim.state.clear()
            self._beat_head.to(self.device)
            return

        # 从一小批数据推断 num_classes（不依赖模型 meta）
        # self.model.eval()
        # try:
        #     first_batch = next(iter(self.trainloader))
        # except StopIteration:
        #     raise RuntimeError("[FedBEAT-Client] Empty trainloader; cannot init BEAT head.")
        # x0 = first_batch[0].to(self.device)
        # with torch.no_grad():
        #     logits = self.model(x0)
        #     if logits.dim() != 2:
        #         logits = logits.view(logits.size(0), -1)
        #     num_classes = int(logits.size(1))
        was_training = self.model.training
        self.model.eval()
        first_batch = next(iter(self.trainloader))
        x0 = first_batch[0].to(self.device)
        with torch.no_grad():
            logits = self.model(x0)
            if logits.dim() != 2:
                logits = logits.view(logits.size(0), -1)
            num_classes = int(logits.size(1))
        self.model.train(was_training)  # ← 恢复

        # 初始化 BEAT 残差头（logits→residual logits）及其优化器
        self._beat_head = BEATModule(num_classes=num_classes, hidden=128, use_dropout=True).to(self.device)
        self._beat_optim = torch.optim.Adam(self._beat_head.parameters(), lr=1e-3)

        logger.info(
            "[FedBEAT-Client %s] Initialized BEAT head: C=%d, hidden=128, optim=Adam(lr=1e-3)",
            getattr(self, "client_id", "?"), num_classes
        )

    # ----------------------------------------------------------------------
    # 关键修复：清理 BN 的 buffers，防止 'leaf Variable' 报错
    # ----------------------------------------------------------------------
    @staticmethod
    def _sanitize_batchnorm_buffers(model: nn.Module) -> None:
        """
        某些环境/阶段切换后，BN 的 num_batches_tracked 可能被误标记为 requires_grad=True。
        这里统一断开其计算图并置回 False；对 running_mean/var 也同样处理以确保万无一失。
        """
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    # num_batches_tracked
                    if hasattr(m, "num_batches_tracked") and isinstance(m.num_batches_tracked, torch.Tensor):
                        try:
                            m.num_batches_tracked.detach_()
                        except Exception:
                            m.num_batches_tracked = m.num_batches_tracked.detach()
                        m.num_batches_tracked.requires_grad = False
                    # running_mean / running_var
                    if hasattr(m, "running_mean") and isinstance(m.running_mean, torch.Tensor):
                        try:
                            m.running_mean.detach_()
                        except Exception:
                            m.running_mean = m.running_mean.detach()
                        m.running_mean.requires_grad = False
                    if hasattr(m, "running_var") and isinstance(m.running_var, torch.Tensor):
                        try:
                            m.running_var.detach_()
                        except Exception:
                            m.running_var = m.running_var.detach()
                        m.running_var.requires_grad = False
