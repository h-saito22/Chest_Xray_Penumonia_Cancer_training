"""閾値を動かしたときのFN/FPを比べる(ノイズなし vs ノイズ50%、どちらもimagenet正規化)。"""
import numpy as np
import torch
from pathlib import Path
from dataset import get_dataloaders
from models import ResNet_finetuning

MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"

def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            probs.append(torch.softmax(model(images), dim=1)[:, 1])   # 肺炎の確率
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()

def count(probs, labels, th):
    pred = (probs >= th).astype(int)
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    return fn, fp

if __name__ == "__main__":
    _, _, test_loader = get_dataloaders(norm="imagenet")

    results = {}
    for name, filename in [("ノイズなし", "resnet_finetuning_imagenet.pth"),
                           ("ノイズ50%", "resnet_finetuning_noise_imagenet.pth")]:
        model, _ = ResNet_finetuning()
        model.load_state_dict(torch.load(MODEL_DIR / filename))
        results[name] = get_probs(model, test_loader)

    # ① 閾値ごとのFN/FP
    print("閾値 | ノイズなし FN/FP | ノイズ50% FN/FP")
    for th in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        a = count(*results["ノイズなし"], th)
        b = count(*results["ノイズ50%"], th)
        print(f"{th:4.2f} | FN={a[0]:3d} FP={a[1]:3d} | FN={b[0]:3d} FP={b[1]:3d}")

    # ② 見逃しをそろえて比較：ノイズなしのモデルで、FNが2件以下になる最大の閾値を探す
    target_fn = count(*results["ノイズ50%"], 0.5)[0]
    probs, labels = results["ノイズなし"]
    for th in np.arange(0.50, 0.0, -0.01):
        fn, fp = count(probs, labels, th)
        if fn <= target_fn:
            print(f"\n見逃しを{target_fn}件以下にそろえると → ノイズなし: 閾値{th:.2f}で FN={fn}, FP={fp}"
                  f" / ノイズ50%: 閾値0.50で FN={target_fn}, FP={count(*results['ノイズ50%'], 0.5)[1]}")
            break