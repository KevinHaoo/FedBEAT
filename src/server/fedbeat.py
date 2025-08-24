# file: src/server/fedbeat.py
import argparse
import logging
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from .fedavg import FedAvgServer
from src.client.fedbeat import FedBEATClient

# 用你已经实现好的四个 MAIR 适配函数（不要改名）
from src.utils.robust.attacks.blackboxbench.wrappers import (
    run_attack_square,
    run_attack_spsa,
    run_attack_onepixel,
    run_attack_pixle,
)
from ..utils.constants import DATA_SHAPE, DATA_STD, DATA_MEAN
from ..utils.robust.wrappers import AttackModelWrapper

# --- RS / Denoise baselines ---
from src.utils.robust.rs.smoothing import randomized_smoothing
from src.utils.robust.denoise.median import median_filter
from src.utils.robust.denoise.tiny_ae import TinyAE

# --- GAP: Generative Adversarial Purification ---
from src.utils.robust.gap.purify import build_gap, load_gap, GapPurifier

logger = logging.getLogger(__name__)

# === [PATCH] FedBEAT 评测工具（新增） ===
def _attack_runner_by_name(name):
    name = str(name).lower()
    table = {
        "square":   run_attack_square,
        "spsa":     run_attack_spsa,
        "onepixel": run_attack_onepixel,
        "pixle":    run_attack_pixle,
    }
    if name not in table:
        raise ValueError(f"[FedBEAT] Unknown attack: {name}")
    return table[name]
# === [PATCH END] ===


class FedBEATServer(FedAvgServer):
    """
    FedBEAT 的 server 端：
      - 训练/聚合逻辑继承 FedAvgServer（客户端已执行 BEAT/DBFAT）
      - 新增：server 侧集中黑盒评测（可选），并把 adv_acc_* / robust_gap_* 写入 metrics.csv
    """
    algorithm_name = "FedBEAT"
    client_cls = FedBEATClient

    @staticmethod
    def get_hyperparams(argv=None):
        if argv is None:
            argv = []
        p = argparse.ArgumentParser(add_help=False)
        # 仅保留与 BEAT 相关的一个超参；FedAvg 阶段仍走 common.local_epoch
        p.add_argument("--epochs_beat", type=int, default=1)
        ns, _ = p.parse_known_args(argv)
        return ns

    def __init__(self, args: DictConfig):
        super().__init__(args)

        fb = dict(args.get("fedbeat", {}))
        atk = dict(fb.get("attacks", {}))

        # ---- 攻击全局参数（训练/评测共享）----
        self._atk_active: str = str(atk.get("active", fb.get("attack", "square"))).lower()
        self._atk_eps: float = float(atk.get("eps", fb.get("eps", 8 / 255)))
        self._atk_steps_default: int = int(atk.get("steps", fb.get("steps", 30)))
        self._atk_budget: int = int(atk.get("budget", fb.get("budget", 10000)))
        self._atk_list: List[Dict] = list(atk.get("list", []))

        # --- MNIST 等灰度数据的 eps 兜底 ---
        ds_name = str(self.args.dataset.name).lower()
        self._atk_eps_eval = self._atk_eps
        if ds_name in ("mnist", "fmnist", "emnist") and self._atk_eps <= 0.1:
            self.logger.log(f"[FedBEAT] eps={self._atk_eps} looks small for {ds_name}; using 0.3 for eval.")
            self._atk_eps_eval = 0.3

        # ---- eval 钩子 ----
        ev = dict(fb.get("eval", {}))
        self.adv_enabled: bool = bool(ev.get("adv_enabled", False))
        self.adv_interval: int = int(ev.get("interval_rounds", -1))
        self.adv_max_eval_examples: int = int(ev.get("max_eval_examples", 128))

        raw_names: List[str] = list(ev.get("attacks_to_eval", [])) or ["active"]
        self._attacks_to_eval: List[str] = self._resolve_attacks_to_eval(raw_names)

        # 步数覆盖（评测可与训练不同）
        self._eval_step_override: Dict[str, int] = {str(k).lower(): int(v) for k, v in ev.get("eval_step_override", {}).items()}

        # 每轮的对抗评测缓存（用于写 CSV）
        self._adv_cache_by_round: Dict[int, Dict[str, float]] = {}

        self._last_adv_round_metrics: Optional[Dict[str, float]] = None  # PATCH: 本轮评测结果

        # Input Denoise
        self._ae = None  # lazy build tiny AE when denoise.kind == 'ae'

        # --- GAP config ---
        gap_cfg = getattr(self.args.fedbeat, "gap", None)
        self._gap_enabled = bool(gap_cfg and getattr(gap_cfg, "enabled", False))
        self._gap_kind = str(getattr(gap_cfg, "kind", "tiny_unet"))
        self._gap_ckpt = str(getattr(gap_cfg, "ckpt", ""))
        self._gap_sigma = float(getattr(gap_cfg, "noise_sigma", 0.0))
        self._gap_tta = int(getattr(gap_cfg, "tta_n", 1))
        self._gap = None  # generator
        self._gap_purifier = None  # wrapper

    # ----------------- 工具函数 ----------------- #
    def _build_gap_if_needed(self, in_ch: int):
        if self._gap is not None and self._gap_purifier is not None:
            return
        self._gap = build_gap(self._gap_kind, in_ch=in_ch, base=32).to(self.device).eval()
        # 可选加载权重
        try:
            load_gap(self._gap_ckpt, self._gap, self.device)
            self.logger.log(f"[FedBEAT-Server] GAP generator ready (kind={self._gap_kind}, ckpt='{self._gap_ckpt}').")
        except Exception as e:
            self.logger.log(f"[FedBEAT-Server] GAP ckpt load failed ({e}); use random-initialized generator.")
        self._gap_purifier = GapPurifier(self._gap, noise_sigma=self._gap_sigma, tta_n=self._gap_tta).to(self.device).eval()


    def _build_denoiser_if_needed(self, in_ch: int):
        """Lazy 构建 TinyAE（只在 denoise.kind == 'ae' 时用到）。"""
        if self._ae is not None:
            return
        base = 16
        self._ae = TinyAE(in_ch=in_ch, base=base).to(self.device).eval()
        ae_ckpt = getattr(self.args.fedbeat.denoise, "ae_ckpt", "")
        if ae_ckpt:
            try:
                state = torch.load(ae_ckpt, map_location=self.device)
                self._ae.load_state_dict(state, strict=False)
                self.logger.log(f"[FedBEAT-Server] TinyAE loaded from {ae_ckpt}.")
            except Exception as e:
                self.logger.log(f"[FedBEAT-Server] TinyAE ckpt load failed ({e}), use randomly-initialized AE.")

    @torch.no_grad()
    def _forward_with_defense(self, mdl, x: torch.Tensor) -> torch.Tensor:
        """
        统一的“带黑盒防御”的前向输出 logits：
        - 若 RS 打开：随机平滑（n 次加噪前向求均值 logits）
        - 若 Denoise 打开：先净化（median / AE），再前向
        - 都没开：直接 mdl(x)
        """
        rs_cfg = getattr(self.args.fedbeat, "rs", None)
        dn_cfg = getattr(self.args.fedbeat, "denoise", None)

        use_rs = bool(rs_cfg and getattr(rs_cfg, "enabled", False))
        use_gap = bool(self._gap_enabled)
        use_dn = bool(dn_cfg and getattr(dn_cfg, "enabled", False))

        # 二选一：若两者都开，优先 RS（与我们“统计/净化 baseline”思路一致，可按需调整）
        if use_rs:
            sigma = float(getattr(rs_cfg, "sigma", 0.25))
            n = int(getattr(rs_cfg, "n_samples", 1))
            # randomized_smoothing 要求给 "model_fn"
            return randomized_smoothing(lambda z: mdl(z), x, sigma=sigma, n=n)

        if use_gap:
            x01 = self._to_01(x)
            self._build_gap_if_needed(in_ch=int(x01.size(1)))
            x01_pur = self._gap_purifier(x01)  # 0..1
            xn = self._from_01(x01_pur)
            return mdl(xn)

        if use_dn:
            kind = str(getattr(dn_cfg, "kind", "median"))
            if kind == "median":
                k = int(getattr(dn_cfg, "median_kernel", 3))
                x_den = median_filter(x, kernel=k)
                return mdl(x_den)
            elif kind == "ae":
                in_ch = x.size(1)
                self._build_denoiser_if_needed(in_ch)
                x_den = self._ae(x)
                return mdl(x_den)
            else:
                # 预留 TV/其他；当前没实现则直接透传
                return mdl(x)

        # 默认：无防御
        return mdl(x)

    def _run_adv_eval_on_global(self, round_idx: int) -> dict:
        """
        在 server 侧评估：Acln（干净准确率，按子集）/ Arob（同一子集、对抗后），
        并返回要写入 CSV 的列字典。
        - 攻击在 [0,1] 图像域进行（eps/steps 都按 0-1 语义）
        - 模型前向时再做 normalize
        - ΔL∞ 也在 [0,1] 上统计
        """
        cfg = self.args.fedbeat.eval
        ds_name = self.args.dataset.name.lower()
        device = self.device

        # 只抽一小撮做 server 评测，加速
        max_n = int(cfg.get("max_eval_examples", 128))
        attacks = cfg.get("attacks_to_eval", ["active"])
        if isinstance(attacks, str):
            attacks = [attacks]

        self.model.eval()
        # 统一用 AttackModelWrapper：输入期待 [0,1]，内部做 normalize 再调 self.model
        bb = AttackModelWrapper(self.model).to(device)
        bb.eval()

        # 取一批集中测试数据（你现有的 server 侧 loader 名叫 self.test_loader）
        xs, ys = [], []
        with torch.no_grad():
            for xb, yb in self.testloader:
                xs.append(xb)
                ys.append(yb)
                if sum(t.shape[0] for t in xs) >= max_n:
                    break
        x = torch.cat(xs, 0)[:max_n].to(device)
        y = torch.cat(ys, 0)[:max_n].to(device)

        # 注意：x 当前是“已 normalize”张量，需要先转回 [0,1]
        x01 = self._to_01(x)

        # 先在干净子集上过一遍：只统计“本来就预测正确”的部分
        with torch.no_grad():
            pred_clean = bb(x01).argmax(1)
            mask = pred_clean.eq(y)
            n_clean = int(mask.sum().item())

        if n_clean == 0:
            # 没有可评样本，给 0 并返回
            msg = f"[FedBEAT-Server] Round {round_idx} Acln=0 (no correct clean samples)"
            self.logger.info(msg)
            out = {"server_clean_acc": 0.0}
            for atk in attacks:
                out[f"server_adv_acc__{atk}"] = 0.0
                out[f"robust_gap__{atk}"] = 0.0
            out["server_adv_acc_avg"] = 0.0
            out["robust_gap_avg"] = 0.0
            return out

        x01_clean = x01[mask]
        y_clean = y[mask]
        # Acln 基于“干净正确子集”的准确率（等于 100%）
        # 若你更偏好“全量 testset 的准确率”，把 mask 去掉即可 —— 但那样 Arob 会被 clean-mis 样本“稀释”
        Acln = 100.0  # 因为 pred_clean==y 的子集上，干净准确率必然 100%

        # 逐攻击评测
        acc_list = []
        extra_cols = {"server_clean_acc": float(Acln)}

        # 读取 eps/steps（默认从 fedbeat.attacks）
        atk_cfg = self.args.fedbeat.attacks
        eps_01 = float(atk_cfg.get("eps", 0.3))  # 0-1 域 eps
        budget = int(atk_cfg.get("budget", 1000))
        override = dict(getattr(cfg, "eval_step_override", {}))  # e.g., square: 30

        for name in attacks:
            # 取 steps 覆盖
            steps = int(override.get(name, atk_cfg.get("steps", 30)))

            # === 调 wrappers：传入 AttackModelWrapper（吃 [0,1]），也传 ds_name 让边界=0..1 ===
            try:
                if name == "square":
                    x_adv01 = run_attack_square(bb, x01_clean, y_clean, eps=eps_01, steps=steps, dataset=ds_name)
                elif name == "spsa":
                    x_adv01 = run_attack_spsa(bb, x01_clean, y_clean, eps=eps_01, steps=steps, dataset=ds_name)
                elif name == "onepixel":
                    x_adv01 = run_attack_onepixel(bb, x01_clean, y_clean, steps=steps, dataset=ds_name)
                elif name == "pixle":
                    x_adv01 = run_attack_pixle(bb, x01_clean, y_clean, steps=steps, dataset=ds_name)
                else:
                    self.logger.warning(f"[FedBEAT-Server] Unknown attack '{name}', skip.")
                    continue
            except Exception as e:
                self.logger.warn(f"[FedBEAT-Server] attack {name} failed: {e}")
                continue

            # 0-1 域上的 L∞（更有意义），再算 Arob（同一子集）
            with torch.no_grad():
                delta_linf = (x_adv01 - x01_clean).abs().flatten(1).max(1).mean().item()
                pred_adv = bb(x_adv01).argmax(1)
                Arob = 100.0 * pred_adv.eq(y_clean).float().mean().item()

            gap = max(0.0, Acln - Arob)
            self.logger.info(
                f"[FedBEAT-Server] Round {round_idx} {name}: Arob={Arob:.2f}, gap={gap:.2f}, ΔL∞(01)={delta_linf:.4f}")

            extra_cols[f"server_adv_acc__{name}"] = float(Arob)
            extra_cols[f"robust_gap__{name}"] = float(gap)
            acc_list.append(Arob)

        # 平均
        if acc_list:
            Arob_avg = float(np.mean(acc_list))
            gap_avg = max(0.0, Acln - Arob_avg)
        else:
            Arob_avg = 0.0
            gap_avg = 0.0
        extra_cols["server_adv_acc_avg"] = Arob_avg
        extra_cols["robust_gap_avg"] = gap_avg

        return extra_cols

    # === [ADD] cache helpers for adv metrics ===
    def _ensure_adv_for_round(self, r: int):
        if not hasattr(self, "_adv_cache"):
            self._adv_cache = {}
        if r in self._adv_cache:
            return
        # 真正去跑一次评测并把结果丢进 cache
        self._adv_cache[r] = self._run_adv_eval_on_global(round_idx=r)

    def _pop_adv_metrics_for_round(self, r: int):
        if not hasattr(self, "_adv_cache"):
            return {}
        return self._adv_cache.pop(r, {})

    # === [ADD] normalization helpers ===
    def _norm_tensors(self):
        ds = self.args.dataset.name.lower()
        C, H, W = DATA_SHAPE[ds]
        mean = torch.tensor(DATA_MEAN[ds], device=self.device).view(1, C, 1, 1)
        std = torch.tensor(DATA_STD[ds], device=self.device).view(1, C, 1, 1)
        return mean, std

    def _to_01(self, x_norm):
        # x_norm: normalized tensor; return clamped [0,1]
        mean, std = self._norm_tensors()
        return torch.clamp(x_norm * std + mean, 0.0, 1.0)

    def _from_01(self, x01):
        # x01: [0,1]; return normalized
        mean, std = self._norm_tensors()
        return (x01 - mean) / std

    # --- NEW: 每一轮新鲜取 server 评测子集（避免 DataLoader 迭代器被吃光） ---
    def _take_eval_subset(self, max_n: int):
        """
        返回 (x_eval, y_eval) ，已放到 self.device。
        每轮用固定种子 + 轮次扰动，保证“可复现且轮轮不同”。
        """
        loader = getattr(self, "testloader", None)
        assert loader is not None, "Server test loader not found."

        base_ds = getattr(loader, "dataset", None)
        assert base_ds is not None, "Server test dataset not found."

        n_total = len(base_ds)
        if n_total == 0:
            raise RuntimeError("Empty server test dataset.")

        n = min(int(max_n), n_total)

        # 轮次相关的可复现采样
        g = torch.Generator()
        g.manual_seed(1337 + int(self.current_epoch))  # 每轮不同，但可复现
        idx = torch.randperm(n_total, generator=g)[:n].tolist()

        xs, ys = [], []
        for i in idx:
            x, y = base_ds[i]
            xs.append(x.unsqueeze(0))
            ys.append(torch.tensor([y], dtype=torch.long))

        x = torch.cat(xs, dim=0).to(self.device, non_blocking=True)
        y = torch.cat(ys, dim=0).to(self.device, non_blocking=True)

        # 便于诊断：每轮打印一下采样规模
        self.logger.log(f"[FedBEAT-Server] Round {int(self.current_epoch) + 1} eval subset size = {x.size(0)}")
        return x, y

    def _resolve_attacks_to_eval(self, names_in: List[str]) -> List[str]:
        out, seen = [], set()
        for n in names_in:
            n = str(n).lower()
            if n == "active":
                n = self._atk_active
            if n not in seen:
                seen.add(n); out.append(n)
        return out

    def _get_attack_runner(self, name: str):
        name = str(name).lower()
        table = {
            "square": run_attack_square,
            "spsa": run_attack_spsa,
            "onepixel": run_attack_onepixel,
            "pixle": run_attack_pixle,
        }
        if name not in table:
            raise ValueError(f"[FedBEAT] Unknown attack: {name}")
        return table[name]

    # ----------------- 对抗评测 ----------------- #
    @torch.no_grad()
    def _eval_adv_on_global(self, atk_name: str) -> float:
        # PATCH: 固定评测随机性（攻击/随机块/采样等）
        try:
            seed = int(getattr(self.args.common, "seed", 42))
        except Exception:
            seed = 42
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 需要 server 侧 centralized 测试集
        testloader = getattr(self, "testloader", None)
        if testloader is None:
            logger.warning("[FedBEAT] No server-side testloader, skip adv eval.")
            return float("nan")

        # 使用聚合后的全局模型；注意：server 模型没有 BEAT 头，满足“黑盒仅 logits”
        model = deepcopy(self.model).to(self.device).eval()

        def model_fn(inp: torch.Tensor) -> torch.Tensor:
            return model(inp)

        # steps 优先用 eval 覆盖
        steps = int(self._eval_step_override.get(atk_name, self._atk_steps_default))
        budget = max(steps, 1)
        dataset_name = str(self.args.dataset.name)

        run = self._get_attack_runner(atk_name)

        seen, correct = 0, 0
        for xb, yb in testloader:
            if seen >= self.adv_max_eval_examples:
                break
            xb, yb = xb.to(self.device), yb.to(self.device)
            x_adv = run(
                model_fn, xb, yb,
                eps=self._atk_eps,
                steps=steps,
                budget=budget,
                dataset=dataset_name,
            )
            pred = model(x_adv).argmax(1)
            correct += (pred == yb).sum().item()
            seen += yb.size(0)

        acc = 100.0 * correct / max(1, seen)
        logger.info(f"[FedBEAT] AdvEval {atk_name}: {acc:.2f}% on {seen} ex (steps={steps})")
        self.logger.log(f"[FedBEAT] AdvEval {atk_name}: {acc:.2f}% on {seen} ex (steps={steps})")
        return acc

    # ----------------- 训练一轮（挂钩在父类之后） ----------------- #
    def train_one_round(self):
        """
        一轮流程：客户端训练 -> 聚合 -> （可选）server & client 对抗评测 -> 写 CSV。
        注意：这里不调用 super().train_one_round()，以便拿到 client_packages 做本地鲁棒均值。
        """
        # 1) 客户端本地训练（沿用父类调用）
        client_packages = self.trainer.train()

        # 2) 统计上传字节（与 FedAvg 一致）
        if len(client_packages) > 0:
            bytes_list = [pkg.get("upload_bytes", 0) for pkg in client_packages.values()]
            self.round_upload_bytes.append(float(np.mean(bytes_list)))
        else:
            self.round_upload_bytes.append(0.0)

        # 3) 聚合为全局
        self.aggregate_client_updates(client_packages)

        # 4) 写基础 CSV（accuracy_before/after、耗时等）
        # super().save_metrics_stats()

        # 5) 条件触发：server 侧对抗评测（集中测试集，聚合后的全局模型）
        r = int(self.current_epoch)  # 与 CSV 的 epoch 行对应（1-based）
        self._last_adv_round_metrics = None
        round_metrics: Dict[str, float] = {}

        if bool(getattr(self.args.fedbeat.eval, "adv_enabled", False)) \
                and (int(self.current_epoch) + 1) % int(getattr(self.args.fedbeat.eval, "interval_rounds", 1)) == 0:
            epoch1 = int(self.current_epoch) + 1
            # x_eval, y_eval = self._take_eval_subset(int(getattr(self.args.fedbeat.eval, "max_eval_examples", 64)))
            #
            # mdl = self.model.to(self.device).eval()
            #
            # # 关键：把“模型”和“数据”都放回 [0,1] 口径
            # bb = AttackModelWrapper(mdl).to(self.device).eval()  # 吃 [0,1]，内部 normalize 再 forward
            # x01 = self._to_01(x_eval)  # normalize -> [0,1]
            #
            # # Acln：同一批干净样本、在 [0,1] 域用 bb 计算
            # with torch.no_grad():
            #     # pred_clean = bb(x01).argmax(1)
            #     # server_clean_acc = (pred_clean == y_eval).float().mean().item() * 100.0
            #     logits_clean = self._forward_with_defense(mdl, x_eval)
            #     pred_clean = logits_clean.argmax(1)
            #     server_clean_acc = (pred_clean == y_eval).float().mean().item() * 100.0
            # round_metrics = {"server_clean_acc": server_clean_acc}
            # self.logger.log(f"[FedBEAT-Server] Round {epoch1} Acln={server_clean_acc:.2f}")
            #
            # # Arob：对“同一批样本”施加攻击（wrappers 也都在 [0,1] 工作），再用 bb 评测
            # atk_eps = float(self._atk_eps_eval)  # MNIST 时会兜底到 0.3
            # atk_steps_default = int(getattr(self.args.fedbeat.attacks, "steps", 30))
            # step_override = dict(getattr(self.args.fedbeat.eval, "eval_step_override", {}))
            # attacks_to_eval = getattr(self.args.fedbeat.eval, "attacks_to_eval", []) or \
            #                   [getattr(self.args.fedbeat.attacks, "active", "square")]
            #
            # adv_acc_values, per_attack_gaps = [], []
            # for atk in attacks_to_eval:
            #     steps = int(step_override.get(atk, atk_steps_default))
            #     runner = _attack_runner_by_name(atk)
            #
            #     x_adv01 = runner(bb, x01, y_eval, eps=atk_eps, steps=steps,
            #                      budget=max(steps, 1), dataset=str(self.args.dataset.name))
            #
            #     # 诊断 + 评测
            #     with torch.no_grad():
            #         delta_linf = (x_adv01 - x01).abs().flatten(1).max(1).values.mean().item()
            #         # pred_adv = bb(x_adv01).argmax(1)
            #         # adv_acc = (pred_adv == y_eval).float().mean().item() * 100.0
            #         logits_adv = self._forward_with_defense(mdl, x_adv01)
            #         pred_adv = logits_adv.argmax(1)
            #         adv_acc = (pred_adv == y_eval).float().mean().item() * 100.0
            #     gap = max(0.0, server_clean_acc - adv_acc)
            #
            #     round_metrics[f"server_adv_acc__{atk}"] = adv_acc
            #     round_metrics[f"robust_gap__{atk}"] = gap
            #     adv_acc_values.append(adv_acc)
            #     per_attack_gaps.append(gap)
            #     self.logger.log(
            #         f"[FedBEAT-Server] Round {epoch1} {atk}: Arob={adv_acc:.2f}, gap={gap:.2f}, ΔL∞(01)={delta_linf:.4f}")
            #
            # # 平均
            # if adv_acc_values:
            #     round_metrics["server_adv_acc_avg"] = float(np.mean(adv_acc_values))
            # if per_attack_gaps:
            #     round_metrics["robust_gap_avg"] = float(np.mean(per_attack_gaps))
            #
            # self._adv_cache_by_round[epoch1] = round_metrics.copy()
            # self._last_adv_round_metrics = round_metrics.copy()

            # 1) 取子集（每轮新取一次），并统一模型 device
            x_eval, y_eval = self._take_eval_subset(int(getattr(self.args.fedbeat.eval, "max_eval_examples", 64)))
            mdl = self.model.to(self.device).eval()

            # 2) 计算 Acln（在同一个随机采样子集上）
            with torch.no_grad():
                # pred_clean = mdl(x_eval).argmax(1)
                # server_clean_acc = (pred_clean == y_eval).float().mean().item() * 100.0
                logits_clean = self._forward_with_defense(mdl, x_eval)
                pred_clean = logits_clean.argmax(1)
                server_clean_acc = (pred_clean == y_eval).float().mean().item() * 100.0
            round_metrics = {"server_clean_acc": server_clean_acc}

            epoch1 = int(self.current_epoch) + 1
            self.logger.log(f"[FedBEAT-Server] Round {epoch1} Acln={server_clean_acc:.2f}")

            # 2) Arob（对同一子集施加攻击）
            adv_acc_values, per_attack_gaps = [], []
            attacks_to_eval = getattr(self.args.fedbeat.eval, "attacks_to_eval", [])
            if not attacks_to_eval:
                attacks_to_eval = [getattr(self.args.fedbeat.attacks, "active", "square")]

            atk_eps = float(self._atk_eps_eval)  # MNIST 已在 __init__ 做 0.3 兜底
            atk_steps_default = int(getattr(self.args.fedbeat.attacks, "steps", 30))
            step_override = dict(getattr(self.args.fedbeat.eval, "eval_step_override", {}))

            for atk in attacks_to_eval:
                steps = int(step_override.get(atk, atk_steps_default))
                runner = _attack_runner_by_name(atk)

                # 第一次尝试
                x_adv = runner(
                    lambda z: mdl(z),  # 只暴露 logits 的黑盒前向
                    x_eval, y_eval,
                    eps=atk_eps,
                    steps=steps,
                    budget=max(steps, 1),
                    dataset=str(self.args.dataset.name),
                )

                # 诊断：看是否真扰动了
                delta_linf = (x_adv - x_eval).detach().abs().view(x_eval.size(0), -1).max(dim=1)[0].mean().item()
                if delta_linf < 1e-7:
                    self.logger.log(f"[FedBEAT-Server] {atk}: ΔL∞≈0 @steps={steps} → escalate to {steps * 3}.")
                    x_adv = runner(
                        lambda z: mdl(z),
                        x_eval, y_eval,
                        eps=atk_eps,
                        steps=steps * 3,
                        budget=max(steps * 3, 1),
                        dataset=str(self.args.dataset.name),
                    )
                    delta_linf = (x_adv - x_eval).detach().abs().view(x_eval.size(0), -1).max(dim=1)[0].mean().item()

                if delta_linf < 1e-7 and atk != "spsa":
                    self.logger.log(f"[FedBEAT-Server] {atk}: still ΔL∞≈0 → fallback to SPSA.")
                    x_adv = _attack_runner_by_name("spsa")(
                        lambda z: mdl(z),
                        x_eval, y_eval,
                        eps=atk_eps,
                        steps=max(steps, 200),
                        budget=max(steps, 200),
                        dataset=str(self.args.dataset.name),
                    )
                    delta_linf = (x_adv - x_eval).detach().abs().view(x_eval.size(0), -1).max(dim=1)[0].mean().item()

                with torch.no_grad():
                    # pred_adv = mdl(x_adv).argmax(1)
                    # adv_acc = (pred_adv == y_eval).float().mean().item() * 100.0
                    logits_adv = self._forward_with_defense(mdl, x_adv)
                    pred_adv = logits_adv.argmax(1)
                    adv_acc = (pred_adv == y_eval).float().mean().item() * 100.0

                gap = max(0.0, server_clean_acc - adv_acc)
                round_metrics[f"server_adv_acc__{atk}"] = adv_acc
                round_metrics[f"robust_gap__{atk}"] = gap
                adv_acc_values.append(adv_acc)
                per_attack_gaps.append(gap)

                self.logger.log(
                    f"[FedBEAT-Server] Round {epoch1} {atk}: Arob={adv_acc:.2f}, gap={gap:.2f}, ΔL∞={delta_linf:.4f}"
                )

            if adv_acc_values:
                round_metrics["server_adv_acc_avg"] = float(np.mean(adv_acc_values))
            if per_attack_gaps:
                round_metrics["robust_gap_avg"] = float(np.mean(per_attack_gaps))

            # 3) 缓存，等待下一次 save_metrics_stats() 精准写回同一轮
            self._adv_cache_by_round[epoch1] = round_metrics.copy()
            self._last_adv_round_metrics = round_metrics.copy()

            # 6) 评测结束可选把模型放回 CPU，避免后续序列化开销出错
            self.model.to(torch.device("cpu"))

        # 6) 合并到 CSV（放到同一轮行）
        # self.save_metrics_stats()  # 调我们覆写的合并函数

    # ----------------- 写 CSV：追加 adv_acc_* / robust_gap_* ----------------- #
    def save_metrics_stats(self, *args, **kwargs):
        """
        先让父类把基础四列（before/after x train/val/test）写到 metrics.csv，
        再把我们训练期间缓存的 Acln/Arob/gap（每轮）补写进去，避免“首轮缺失/错位”。
        """
        # 1) 父类先写基础列
        super().save_metrics_stats(*args, **kwargs)

        # 2) 未开启 adv 评测则不追加
        fe = getattr(self.args, "fedbeat", {}).get("eval", {})
        if not bool(fe.get("adv_enabled", False)):
            return

        # 3) 读回 CSV，逐轮补列
        csv_path = self.output_dir / "metrics.csv"
        if not csv_path.exists():
            return

        import pandas as pd
        df = pd.read_csv(csv_path, index_col="epoch")

        # _adv_cache_by_round: {round_idx: {"server_clean_acc":..., "server_adv_acc__square":..., ...}}
        if hasattr(self, "_adv_cache_by_round"):
            for r, cols in sorted(self._adv_cache_by_round.items()):
                # 确保行存在
                if r-1 not in df.index:
                    # 若父类没有这轮（极少见），扩一行
                    df.loc[r-1] = float("nan")
                # 逐列写入
                for k, v in cols.items():
                    if k not in df.columns:
                        df[k] = float("nan")
                    df.at[r-1, k] = v

        # 4) 回写
        df.sort_index(inplace=True)
        df.to_csv(csv_path, index=True, index_label="epoch")


