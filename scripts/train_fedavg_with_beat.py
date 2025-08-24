# -*- coding: utf-8 -*-
# file: scripts/train_fedavg_with_beat.py
from __future__ import annotations
import argparse, os, sys, shlex, subprocess
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "main.py").exists() and (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Cannot locate project root (expect main.py + config/ + src/).")

def _to_255_frac(v: str) -> str:
    v = v.strip()
    if "/255" in v:
        num = float(v.replace("/255", "").strip())
        return f"{num/255.0:.10f}"
    float(v)  # validate
    return v

def build_overrides(a: argparse.Namespace) -> list[str]:
    eps_txt = _to_255_frac(a.eps)
    ov = [
        # 方法与数据
        "method=fedbeat",
        f"dataset.name={a.dataset}",
        f"model.name={a.model}",
        # 通用训练（FedAvg 阶段沿用 common.local_epoch）
        f"common.global_epoch={a.global_epoch}",
        f"common.local_epoch={a.local_epoch}",
        f"common.join_ratio={a.join_ratio}",
        f"common.batch_size={a.batch_size}",
        f"common.seed={a.seed}",
        # 优化器
        f"optimizer.name={a.optimizer}",
        f"optimizer.lr={a.lr}",
        # 评测/保存
        f"common.test.client.interval={a.test_client_interval}",
        f"common.test.server.interval={a.test_server_interval}",
        f"common.save_model={'true' if a.save_model else 'false'}",
        f"common.save_metrics={'true' if a.save_metrics else 'false'}",
        # fedbeat：BEAT 阶段本地 epoch + KL 权重
        f"fedbeat.epochs_beat={a.epochs_beat}",
        f"fedbeat.loss.kl.beta={a.beta}",
        # fedbeat：黑盒攻击（复数 attacks）
        f"fedbeat.attacks.active={a.active}",
        f"fedbeat.attacks.eps={eps_txt}",
        f"fedbeat.attacks.steps={a.steps}",
        f"fedbeat.attacks.budget={a.budget}",
    ]
    if a.exp_name:
        ov.append(f'common.monitor="{a.exp_name}"')
    return ov

def main():
    ap = argparse.ArgumentParser(description="Train FedAvg + BEAT (+ DBFAT) via FL-bench main.py")
    # 基础
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--model", default="lenet5")
    # 通用训练
    ap.add_argument("--global-epoch", type=int, default=10)
    ap.add_argument("--local-epoch", type=int, default=1, help="FedAvg阶段本地epoch（复用common.local_epoch）")
    ap.add_argument("--join-ratio", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    # 黑盒攻击（复数层）
    ap.add_argument("--active", default="square", choices=["square", "spsa", "onepixel", "pixle"])
    ap.add_argument("--eps", default="8/255")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--budget", type=int, default=10000)
    # DBFAT/BEAT 阶段
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--epochs-beat", type=int, default=1)
    # 评测/保存
    ap.add_argument("--test-client-interval", type=int, default=1)
    ap.add_argument("--test-server-interval", type=int, default=1)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-metrics", action="store_true", default=True)
    # 其他
    ap.add_argument("--exp-name", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", default=True, action="store_true")
    a = ap.parse_args()

    root = _find_project_root(Path(__file__).parent)
    main_py = root / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Cannot find main.py at {main_py}")

    overrides = build_overrides(a)
    cmd = [sys.executable, str(main_py)] + overrides

    print("\n[Hydra overrides]")
    print(" \\\n  ".join(shlex.quote(x) for x in overrides))
    print("\n[Run command]")
    print(" ".join(shlex.quote(x) for x in cmd))

    if a.run and not a.dry_run:
        rc = subprocess.run(cmd, cwd=str(root), env=os.environ.copy()).returncode
        sys.exit(rc)
    else:
        print("\n(dry-run) Not executing. Add --run to actually start training.")

if __name__ == "__main__":
    main()
