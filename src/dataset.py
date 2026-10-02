"""胸部X線 肺炎分類プロジェクト用のデータセット関連処理をまとめたモジュール。

- 画像パスとラベルの一覧をDataFrameとして作成 (make_dataset)
- ピクセル値のヒストグラムからmean/stdを計算 (compute_norm_stats)
- train/val/test用のtransformを作成 (get_transforms)
- PyTorchのDataset実装 (PneumoniaDataset)
- train/val/test用のDataLoaderをまとめて作成 (get_dataloaders)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# このファイル (src/dataset.py) から見たデータフォルダの場所。
# src/ の親 (リポジトリルート) から
# 01_chest-xray-pneumonia-calssification/notebooks/data を辿る。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT /  "notebooks" / "data"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def make_dataset(kind: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """train/val/test の画像パスとラベルの一覧をDataFrameにまとめる。

    kind: "train" / "val" / "test"
    label: NORMAL=0, PNEUMONIA=1
    """
    dfs = []
    for species, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        selected_path = data_dir / kind / species
        paths = [p for p in selected_path.iterdir() if p.is_file()]
        dfs.append(pd.DataFrame({"label": label, "path": paths}))
    return pd.concat(dfs, ignore_index=True)


def _get_pixel_histogram(kind: str, species: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    """指定したkind/speciesの全画像をグレースケールに変換し、
    ピクセル値ヒストグラム(0-255)の合計を返す。
    """
    selected_path = data_dir / kind / species
    histograms = []
    for p in selected_path.iterdir():
        if p.is_file():
            with Image.open(p) as image:
                image = image.convert("L")
                histograms.append(image.histogram())
    histograms = np.asarray(histograms)
    return np.sum(histograms, axis=0)


def compute_norm_stats(kind: str = "train", data_dir: Path = DATA_DIR):
    """指定したkind(通常はtrain)の画像全体から、
    正規化に使うmean/std(0-1スケール)を計算する。
    """
    normal_sum = _get_pixel_histogram(kind, "NORMAL", data_dir)
    pneumonia_sum = _get_pixel_histogram(kind, "PNEUMONIA", data_dir)

    b = np.arange(256)
    a = normal_sum + pneumonia_sum

    mean = np.sum(a * b) / np.sum(a)
    var = np.sum((b - mean) ** 2 * a) / np.sum(a)
    std = var ** 0.5

    norm_mean = mean / 255
    norm_std = std / 255
    return norm_mean, norm_std


def get_transforms(norm_mean: float, norm_std: float):
    """train用(Augmentationあり)とval/test用(リサイズ+正規化のみ)のtransformを作成する。"""
    normalize = transforms.Normalize(
        (norm_mean, norm_mean, norm_mean),
        (norm_std, norm_std, norm_std),
    )

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


class PneumoniaDataset(Dataset):
    """make_dataset()で作ったDataFrame(label, path)から画像を読み込むDataset。"""

    def __init__(self, df: pd.DataFrame, transform):
        super().__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label, path = self.df.iloc[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_dataloaders(batch_size: int = BATCH_SIZE, data_dir: Path = DATA_DIR):
    """train/val/test用のDataLoaderを3つまとめて作成する。"""
    norm_mean, norm_std = compute_norm_stats("train", data_dir)
    train_transform, eval_transform = get_transforms(norm_mean, norm_std)

    train_df = make_dataset("train", data_dir)
    val_df = make_dataset("val", data_dir)
    test_df = make_dataset("test", data_dir)

    train_dataset = PneumoniaDataset(train_df, train_transform)
    val_dataset = PneumoniaDataset(val_df, eval_transform)
    test_dataset = PneumoniaDataset(test_df, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"train batches: {len(train_loader)}")
    print(f"val batches:   {len(val_loader)}")
    print(f"test batches:  {len(test_loader)}")
