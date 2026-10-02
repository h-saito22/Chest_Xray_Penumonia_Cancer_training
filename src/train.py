"""各モデルの学習ループをまとめたモジュール。

いずれの関数も、セットアップ済みの model と optimizer を外から受け取り、
関数の中で作り直すことはしない(model/optimizerの用意は models.py 側の責務)。
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = _PROJECT_ROOT  / "outputs" / "models"


def save_model(model, model_name):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUTS_DIR / f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"モデルを保存しました: {save_path}")


def BaseLine_CNN_train(model, optimizer, dataset, epoch_num=3, model_name=None):
    criterion = nn.CrossEntropyLoss()
    mean_list = []
    for epoch in range(epoch_num):
        loss_list = []
        for images, labels in dataset:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_list.append(loss_value)
        mean_loss = np.mean(loss_list)
        mean_list.append(mean_loss)
        print(f"epoch {epoch}: mean_loss = {mean_loss}")
    if model_name is not None:
        save_model(model, model_name)


def resnet_train(model, optimizer, dataset, epoch_num=3, model_name=None):
    criterion = nn.CrossEntropyLoss()
    TP = FP = FN = TN = 0
    mean_list = []
    for epoch in range(epoch_num):
        loss_list = []
        for images, labels in dataset:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_list.append(loss_value)
            pred_label = torch.argmax(outputs, dim=1)
            TP += ((pred_label == 1) & (labels == 1)).sum().item()
            FP += ((pred_label == 1) & (labels == 0)).sum().item()
            FN += ((pred_label == 0) & (labels == 1)).sum().item()
            TN += ((pred_label == 0) & (labels == 0)).sum().item()
        mean_loss = np.mean(loss_list)
        mean_list.append(mean_loss)
        print(f"epoch {epoch}: mean_loss = {mean_loss}")
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (recall + precision)
    if model_name is not None:
        save_model(model, model_name)
    return accuracy, precision, recall, F1


def efficient_train(model, optimizer, dataset, epoch_num=3, model_name=None):
    criterion = nn.CrossEntropyLoss()
    TP = FP = FN = TN = 0
    mean_list = []
    model.train()
    for epoch in range(epoch_num):
        loss_list = []
        for images, labels in dataset:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_list.append(loss_value)
            pred_label = torch.argmax(outputs, dim=1)
            TP += ((pred_label == 1) & (labels == 1)).sum().item()
            FP += ((pred_label == 1) & (labels == 0)).sum().item()
            FN += ((pred_label == 0) & (labels == 1)).sum().item()
            TN += ((pred_label == 0) & (labels == 0)).sum().item()
        mean_loss = np.mean(loss_list)
        mean_list.append(mean_loss)
        print(f"epoch {epoch}: mean_loss = {mean_loss}")
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (recall + precision)
    if model_name is not None:
        save_model(model, model_name)
    return accuracy, precision, recall, F1
