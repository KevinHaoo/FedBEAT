import numpy as np
import os

# 你的 npz 文件路径
npz_path = "/root/autodl-tmp/FL-bench/data/medmnistS/raw/aaa.npz"   # 改成你的实际文件名
save_dir = "/root/autodl-tmp/FL-bench/data/medmnistS/raw"   # 保存目录，符合 datasets.py 要求

os.makedirs(save_dir, exist_ok=True)

# 加载 npz
data = np.load(npz_path)

# 通常有这几个 key
x_train, y_train = data["train_images"], data["train_labels"]
x_val,   y_val   = data["val_images"],   data["val_labels"]
x_test,  y_test  = data["test_images"],  data["test_labels"]

# 拼接
x_all = np.concatenate([x_train, x_val, x_test], axis=0)
y_all = np.concatenate([y_train, y_val, y_test], axis=0)

# 确保是灰度 (N,H,W)，去掉多余通道
if x_all.ndim == 4 and x_all.shape[-1] == 1:
    x_all = x_all.squeeze(-1)  # [N,H,W]
elif x_all.ndim == 4 and x_all.shape[-1] == 3:
    raise ValueError("当前处理类只支持灰度图，RGB子集要改Dataset类！")

# 标签展平 + 转 0 开始
y_all = y_all.squeeze()
if y_all.min() == 1:  # 如果是 1..11，改成 0..10
    y_all = y_all - 1

# 转换为 uint8 [0,255]，现有代码会自动除以255
if x_all.max() <= 1.0:
    x_all = (x_all * 255).astype(np.uint8)
else:
    x_all = x_all.astype(np.uint8)

# 保存
np.save(os.path.join(save_dir, "xdata.npy"), x_all)
np.save(os.path.join(save_dir, "ydata.npy"), y_all)

print(f"处理完成！共 {x_all.shape[0]} 张图，图像形状 {x_all.shape[1:]}, 标签范围 {y_all.min()}–{y_all.max()}")
print(f"已保存到 {save_dir}/xdata.npy, ydata.npy")
