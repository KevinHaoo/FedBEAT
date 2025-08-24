##!/bin/bash
#
#if [ ! -d "../medmnistS/raw" ]; then
#    mkdir -p ../medmnistS/raw
#fi
#
#cd ../medmnistS/raw
#
#wget https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnist.tar.gz
#
#tar -xzvf medmnist.tar.gz
#
#mv medmnist/* ./
#
#rm -rf medmnist

#!/usr/bin/env bash
set -euo pipefail

# 1) 目标目录与路径（保持与原脚本一致：../medmnistS/raw）
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RAW_DIR="$SCRIPT_DIR/../medmnistS/raw2"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

# 2) 下载 organsmnist.npz（curl 优先，wget 兜底）
URL="https://zenodo.org/records/10519652/files/organsmnist.npz?download=1"
NPZ="organsmnist.npz"
echo ">>> Downloading $NPZ to $RAW_DIR ..."
if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$NPZ"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$NPZ" "$URL"
else
  echo "ERROR: need curl or wget to download."
  exit 1
fi

# 3) 预处理：合并 train/val/test -> xdata.npy / ydata.npy
python3 - <<'PY'
import os, numpy as np
raw_dir = os.getcwd()
npz_path = os.path.join(raw_dir, "organsmnist.npz")
print(f">>> Loading {npz_path}")
data = np.load(npz_path)

x_train, y_train = data["train_images"], data["train_labels"]
x_val,   y_val   = data.get("val_images", None), data.get("val_labels", None)
x_test,  y_test  = data["test_images"],  data["test_labels"]

def cat_nonempty(*arrs):
    arrs = [a for a in arrs if a is not None and isinstance(a, np.ndarray) and a.size > 0]
    return np.concatenate(arrs, axis=0)

x_all = cat_nonempty(x_train, x_val, x_test)
y_all = cat_nonempty(y_train, y_val, y_test)

# 若是灰度 4D(N,H,W,1) 则去掉通道维
if x_all.ndim == 4 and x_all.shape[-1] == 1:
    x_all = x_all.squeeze(-1)  # -> (N,H,W)

# 标签压扁 + 若最小为1则改到从0开始
y_all = y_all.squeeze()
if y_all.min() == 1:
    y_all = y_all - 1

# 像素若在[0,1]则放大到[0,255]，统一存为 uint8
x_all = (x_all * 255).astype(np.uint8) if x_all.max() <= 1.0 else x_all.astype(np.uint8)

np.save(os.path.join(raw_dir, "xdata.npy"), x_all)
np.save(os.path.join(raw_dir, "ydata.npy"), y_all)
print(f">>> Saved xdata.npy ({x_all.shape}) and ydata.npy ({y_all.shape}) to {raw_dir}")
PY

echo "✅ Done: $RAW_DIR/xdata.npy  |  $RAW_DIR/ydata.npy"

