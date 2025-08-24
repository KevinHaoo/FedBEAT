# file: src/utils/robust/gap/purify.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class TinyUNet(nn.Module):
    """
    极简 UNet 风格净化器：输入/输出同形状 [B,C,H,W]，工作在 0..1 像素域
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        C = base
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, C, 3, padding=1), nn.ReLU(), ResBlock(C))
        self.down1 = nn.Conv2d(C, C*2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.ReLU(), ResBlock(C*2))
        self.down2 = nn.Conv2d(C*2, C*4, 4, stride=2, padding=1)
        self.bott = nn.Sequential(nn.ReLU(), ResBlock(C*4))
        self.up1  = nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.ReLU(), ResBlock(C*2))
        self.up2  = nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.ReLU(), ResBlock(C))
        self.out  = nn.Conv2d(C, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b  = self.bott(self.down2(e2))
        d2 = self.dec2(self.up1(b)) + e2
        d1 = self.dec1(self.up2(d2)) + e1
        y  = torch.sigmoid(self.out(d1))  # clamp 到 0..1
        return y

def build_gap(kind: str, in_ch: int, base: int = 32) -> nn.Module:
    kind = str(kind).lower()
    if kind in ("tiny_unet", "unet", "simple"):
        return TinyUNet(in_ch=in_ch, base=base)
    raise ValueError(f"[GAP] unknown generator kind: {kind}")

def load_gap(ckpt_path: str, model: nn.Module, device: torch.device) -> None:
    if not ckpt_path:
        return
    state = torch.load(ckpt_path, map_location=device)
    # 兼容 {'state_dict':...} 或直接权重 dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # 容忍部分层名不对（例如从更大模型裁剪）
        pass

class GapPurifier(nn.Module):
    """
    推理时净化器封装：
    - 输入/输出：0..1 像素域
    - 可选：加微噪并 TTA 多次平均
    """
    def __init__(self, generator: nn.Module, noise_sigma: float = 0.0, tta_n: int = 1):
        super().__init__()
        self.G = generator
        self.noise_sigma = float(noise_sigma)
        self.tta_n = max(1, int(tta_n))

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        outs = []
        for _ in range(self.tta_n):
            z = x01
            if self.noise_sigma > 0:
                z = torch.clamp(z + torch.randn_like(z) * self.noise_sigma, 0.0, 1.0)
            y = self.G(z)
            outs.append(y)
        y = torch.stack(outs, dim=0).mean(0)
        return torch.clamp(y, 0.0, 1.0)

if __name__ == "__main__":
    # 自检
    x = torch.rand(2, 3, 32, 32)
    G = build_gap("tiny_unet", in_ch=3)
    P = GapPurifier(G, noise_sigma=0.0, tta_n=2)
    y = P(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    print("GAP purifier smoke test ok.", y.mean().item())
