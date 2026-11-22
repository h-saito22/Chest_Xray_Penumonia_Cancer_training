"""保存済みモデルのCPU推論時間・サイズを、同じ条件で測定する。"""
import time
import numpy as np
import torch
from pathlib import Path
from dataset import get_dataloaders
from models import get_BaseLine_CNN, get_resnet_frozen, get_efficientnet_setup

MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"
TARGETS = [
    ("Baseline CNN",    get_BaseLine_CNN,       "baseline_dataset.pth"),
    ("ResNet18",        get_resnet_frozen,      "resnet_finetuning_imagenet.pth"),
    ("EfficientNet-B0", get_efficientnet_setup, "efficientnet_imagenet.pth"),
]

def measure(model, batches, n_runs=6, n_warmup=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            for images in batches:
                model(images)
            times.append(time.perf_counter() - start)
    times = times[n_warmup:]
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    torch.set_num_threads(4)   # スレッド数を固定して条件をそろえる
    _, _, test_loader = get_dataloaders(norm="imagenet")
    batches = [images for images, _ in test_loader]   # 先にメモリに読み込む
    n_images = sum(len(b) for b in batches)

    for name, setup_fn, filename in TARGETS:
        model, _ = setup_fn()
        model.load_state_dict(torch.load(MODEL_DIR / filename))
        params = sum(p.numel() for p in model.parameters())
        size_mb = (MODEL_DIR / filename).stat().st_size / 1e6
        mean, std = measure(model, batches)
        print(f"{name:16s} params={params:>11,d}  size={size_mb:6.1f}MB  "
              f"total={mean:6.2f}s ± {std:.2f}s  per_image={mean / n_images * 1000:6.2f}ms")