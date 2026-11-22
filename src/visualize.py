"""最終結果(シード42、学習1回)のモデル比較グラフを作成し、images/に保存する。"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import japanize_matplotlib

SAVE_PATH = Path(__file__).resolve().parent.parent / "images" / "model_comparison.png"

# README 8章の表より(Baselineはdataset、それ以外はimagenetの正規化)
results = [
    {"model": "Baseline CNN",              "accuracy": 0.813, "precision": 0.786, "recall": 0.962, "f1": 0.865, "fn": 15, "fp": 102},
    {"model": "ResNet18\n(frozen)",        "accuracy": 0.861, "precision": 0.826, "recall": 0.985, "f1": 0.898, "fn": 6,  "fp": 81},
    {"model": "ResNet18\n(FineTuning)",    "accuracy": 0.904, "precision": 0.878, "recall": 0.982, "f1": 0.927, "fn": 7,  "fp": 53},
    {"model": "ResNet18\n(FT, noise 50%)", "accuracy": 0.870, "precision": 0.831, "recall": 0.995, "f1": 0.906, "fn": 2,  "fp": 79},
    {"model": "EfficientNet-B0",           "accuracy": 0.886, "precision": 0.910, "recall": 0.908, "f1": 0.909, "fn": 36, "fp": 35},
]

models = [r["model"] for r in results]
x = np.arange(len(results))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左：評価指標
width = 0.2
for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
    ax1.bar(x + (i - 1.5) * width, [r[metric] for r in results], width, label=metric)
ax1.axhline(0.625, color="gray", linestyle="--", linewidth=1, label="all-pneumonia baseline (acc)")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylim(0.5, 1.08)
ax1.set_ylabel("score")
ax1.set_title("Test metrics")
ax1.legend(fontsize=8, loc="upper center", ncol=5)

# 右：間違いの内訳(FN=見逃し、FP=正常を肺炎と判定)
fn = [r["fn"] for r in results]
fp = [r["fp"] for r in results]
ax2.bar(x, fn, label="FN (missed pneumonia)", color="tab:red")
ax2.bar(x, fp, bottom=fn, label="FP (normal judged as pneumonia)", color="tab:orange")
for i, (a, b) in enumerate(zip(fn, fp)):
    ax2.text(i, a + b + 2, str(a + b), ha="center", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylabel("count (test: 624 images)")
ax2.set_title("Errors: FN + FP")
ax2.legend(fontsize=8)

plt.tight_layout()
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SAVE_PATH, dpi=150)
print(f"保存しました: {SAVE_PATH}")
