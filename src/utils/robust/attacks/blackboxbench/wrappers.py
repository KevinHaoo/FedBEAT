# file: src/utils/robust/attacks/blackboxbench/wrappers.py
import os
import sys
from pathlib import Path
from typing import Callable, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

# ---- 1) 绑定 MAIR 路径（按当前 FL-bench 布局自动搜索）----
def _ensure_mair_on_path():
    here = Path(__file__).resolve()
    proj_root = None
    # 先找包含 src/ 与 main.py 的工程根
    for p in here.parents:
        if (p / "src").is_dir() and (p / "config").is_dir() and (p / "main.py").exists():
            proj_root = p
            break
    if proj_root is None:
        # 兜底：退而求其次，找到含 src 的目录
        for p in here.parents:
            if (p / "src").is_dir():
                proj_root = p
                break
    if proj_root is None:
        raise RuntimeError("Cannot locate project root (no 'src' found upwards).")
    cand = proj_root / "third_party" / "MAIR-main"
    env = os.environ.get("MAIR_PATH", None)
    mair_root = Path(env).expanduser().resolve() if env else cand.resolve()
    mair_pkg = mair_root / "mair"
    if not mair_pkg.exists():
        raise ImportError(
            "Cannot import 'mair.attacks'. Please place MAIR-main under "
            "'third_party/MAIR-main' or set MAIR_PATH to MAIR-main."
        )
    if mair_root.as_posix() not in sys.path:
        sys.path.append(mair_root.as_posix())
    return mair_root

_MAIR_ROOT = _ensure_mair_on_path()

# ---- 2) 引入 MAIR 的四个攻击 ----
from mair.attacks.attacks.square import Square
from mair.attacks.attacks.spsa import SPSA
from mair.attacks.attacks.onepixel import OnePixel
from mair.attacks.attacks.pixle import Pixle

# ---- 3) 读取 FL-bench 的数据常量 ----
from src.utils.constants import DATA_MEAN, DATA_STD, DATA_SHAPE


class _AttackableBlackBox(nn.Module):
    """
    包装一个仅有 logits 前向的黑盒函数 model_fn: (Tensor[B,C,H,W]) -> logits[B,num_classes]
    供 MAIR 攻击器统一调用。
    """
    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        n_classes: int,
        input_shape: Tuple[int, int, int],
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        bounds: Tuple[float, float] = (0.0, 1.0),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._model_fn = model_fn
        self.n_classes = int(n_classes)
        self.input_shape = tuple(input_shape)
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)
        self.preprocess: Dict[str, Tuple[float, ...]] = {"mean": self.mean, "std": self.std}
        self.clip_values = tuple(bounds)
        self.bounds = tuple(bounds)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dummy = nn.Parameter(torch.empty(0, device=device), requires_grad=False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def set_device(self, device: torch.device):
        self.to(device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model_fn(x)


def _infer_stats(
    dataset: Optional[str],
    x: torch.Tensor,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, int, int]]:
    if dataset is not None and dataset in DATA_MEAN and dataset in DATA_STD and dataset in DATA_SHAPE:
        mean = tuple(float(m) for m in DATA_MEAN[dataset])
        std = tuple(float(s) for s in DATA_STD[dataset])
        shape = tuple(int(v) for v in DATA_SHAPE[dataset])  # (C,H,W)
    else:
        c, h, w = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        shape = (c, h, w)
        if c == 1:
            mean, std = (0.1307,), (0.3015,)
        else:
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
    return mean, std, shape


def _build_attackable_model(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    dataset: Optional[str] = None,
    n_classes: Optional[int] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
    device: Optional[torch.device] = None,
) -> _AttackableBlackBox:
    if device is None:
        device = x.device if x.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_mean, d_std, d_shape = _infer_stats(dataset, x)
    if mean is not None:
        d_mean = mean
    if std is not None:
        d_std = std
    if n_classes is None:
        with torch.no_grad():
            logits = model_fn(x[:1].to(device))
        n_classes = int(logits.shape[-1])
    return _AttackableBlackBox(
        model_fn=model_fn,
        n_classes=n_classes,
        input_shape=d_shape,
        mean=d_mean,
        std=d_std,
        bounds=bounds,
        device=device,
    )


def run_attack_square(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 1000,
    *,
    dataset: Optional[str] = None,
    n_restarts: int = 1,
    p_init: float = 0.8,
    loss: str = "margin",
    resc_schedule: bool = True,
    seed: int = 0,
    targeted: bool = False,
    device: Optional[torch.device] = None,
    **kw,
) -> torch.Tensor:
    bb = _build_attackable_model(model_fn, x, y, dataset=dataset, device=device)
    atk = Square(
        model=bb,
        norm="Linf",
        eps=float(eps),
        n_queries=int(steps),
        n_restarts=int(n_restarts),
        p_init=float(p_init),
        loss=str(loss),
        resc_schedule=bool(resc_schedule),
        seed=int(seed),
        verbose=bool(kw.get("verbose", False)),
    )
    atk.set_mode_targeted_by_label() if targeted else atk.set_mode_default()
    atk.set_device(bb.device)
    adv = atk(x.to(bb.device), y.to(bb.device))
    return adv.detach()


def run_attack_spsa(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 1000,
    *,
    dataset: Optional[str] = None,
    delta: float = 0.01,
    lr: float = 0.01,
    nb_sample: int = 128,
    max_batch_size: int = 64,
    targeted: bool = False,
    device: Optional[torch.device] = None,
    **kw,
) -> torch.Tensor:
    bb = _build_attackable_model(model_fn, x, y, dataset=dataset, device=device)
    atk = SPSA(
        model=bb,
        eps=float(eps),
        delta=float(delta),
        lr=float(lr),
        nb_iter=int(steps),
        nb_sample=int(nb_sample),
        max_batch_size=int(max_batch_size),
    )
    atk.set_mode_targeted_by_label() if targeted else atk.set_mode_default()
    atk.set_device(bb.device)
    adv = atk(x.to(bb.device), y.to(bb.device))
    return adv.detach()


def run_attack_onepixel(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,  # 未用到（L0）
    steps: int = 30,
    *,
    dataset: Optional[str] = None,
    pixels: int = 1,
    popsize: int = 20,
    inf_batch: int = 256,
    targeted: bool = False,
    device: Optional[torch.device] = None,
    **kw,
) -> torch.Tensor:
    bb = _build_attackable_model(model_fn, x, y, dataset=dataset, device=device)
    atk = OnePixel(
        model=bb,
        pixels=int(pixels),
        steps=int(steps),
        popsize=int(popsize),
        inf_batch=int(inf_batch),
    )
    atk.set_mode_targeted_by_label() if targeted else atk.set_mode_default()
    atk.set_device(bb.device)
    adv = atk(x.to(bb.device), y.to(bb.device))
    return adv.detach()


def run_attack_pixle(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,  # 未用到（L0）
    steps: int = 40,
    *,
    dataset: Optional[str] = None,
    x_dimensions=(2, 8),
    y_dimensions=(2, 8),
    pixel_mapping="random",
    restarts: int = 3,
    update_each_iteration: bool = False,
    targeted: bool = False,
    device: Optional[torch.device] = None,
    **kw,
) -> torch.Tensor:
    bb = _build_attackable_model(model_fn, x, y, dataset=dataset, device=device)
    atk = Pixle(
        model=bb,
        x_dimensions=x_dimensions,
        y_dimensions=y_dimensions,
        pixel_mapping=str(pixel_mapping),
        restarts=int(restarts),
        max_iterations=int(steps),
        update_each_iteration=bool(update_each_iteration),
    )
    atk.set_mode_targeted_by_label() if targeted else atk.set_mode_default()
    atk.set_device(bb.device)
    adv = atk(x.to(bb.device), y.to(bb.device))
    return adv.detach()


# ---------------- 自检 ----------------
def _tiny_logits(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    flat = x.view(B, -1)
    Wlin = torch.randn(flat.size(1), 10, device=flat.device) * 0.01
    b = torch.zeros(10, device=flat.device)
    return flat @ Wlin + b

def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(4, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    tests = {
        "square": lambda: run_attack_square(_tiny_logits, x, y, eps=8/255, steps=5, dataset="mnist"),
        "spsa": lambda: run_attack_spsa(_tiny_logits, x, y, eps=8/255, steps=2, dataset="mnist", nb_sample=8, max_batch_size=4),
        "onepixel": lambda: run_attack_onepixel(_tiny_logits, x, y, eps=0.0, steps=2, dataset="mnist", pixels=1, popsize=4, inf_batch=4),
        "pixle": lambda: run_attack_pixle(_tiny_logits, x, y, eps=0.0, steps=2, dataset="mnist", restarts=1, x_dimensions=(2,4), y_dimensions=(2,4)),
    }
    for name, fn in tests.items():
        try:
            adv = fn()
            assert adv.shape == x.shape
            assert _finite(adv)
            print(f"{name} ok:", adv.shape, adv.min().item(), adv.max().item())
        except Exception as e:
            print(f"{name} failed:", repr(e))
