"""3モデル(Baseline CNN, ResNet18, EfficientNet-B0)の定義・セットアップをまとめたモジュール。

学習ループ(train.py)・評価(evaluate.py)は含めず、
「モデルを作る」「model/optimizerを返す」ところまでがこのファイルの責務。
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv2d = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, padding=1)
        self.Relu = nn.ReLU()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3)
        self.Linear = nn.Linear(in_features=10952, out_features=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.Relu(x)
        x = self.MaxPool2d(x)
        x = self.flatten(x)
        x = self.Linear(x)
        return x

def get_BaseLine_CNN():
    model = BaselineCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model , optimizer

def get_efficientnet_setup():
    """EfficientNet-B0: backboneを凍結し、classifierだけ2クラス用に付け替えたセットアップ。"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(in_features=1280, out_features=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def ResNet_finetuning():
    """ResNet18: layer4だけ解凍し、fcを2クラス用に付け替え、layer4とfcで異なる学習率を使うセットアップ。"""
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in resnet_model.parameters():
        param.requires_grad = False
    for p in resnet_model.layer4.parameters():
        p.requires_grad = True
    resnet_model.fc = nn.Linear(in_features=512, out_features=2)
    optimizer = torch.optim.Adam([
        {"params": resnet_model.layer4.parameters(), "lr": 0.0001},
        {"params": resnet_model.fc.parameters(), "lr": 0.001},
    ])
    return resnet_model, optimizer


def get_resnet_frozen():
    """ResNet18: backboneを全部凍結し、fcだけ2クラス用に付け替えたセットアップ(fine-tuningなし)。"""
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in resnet_model.parameters():
        param.requires_grad = False
    resnet_model.fc = nn.Linear(in_features=512, out_features=2)
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
    return resnet_model, optimizer


if __name__ == "__main__":
    # 学習はせず、各モデルの構造(出力shape)だけ軽く確認する。
    dummy_input = torch.randn(1, 3, 224, 224)

    baseline_model = BaselineCNN()
    out = baseline_model(dummy_input)
    print(f"[BaselineCNN] output shape: {tuple(out.shape)}")

    for name, setup_fn in [
        ("ResNet18 (frozen)", get_resnet_frozen),
        ("ResNet18 (finetuning)", ResNet_finetuning),
        ("EfficientNet-B0", get_efficientnet_setup),
    ]:
        model, optimizer = setup_fn()
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"[{name}] output shape: {tuple(out.shape)}")
