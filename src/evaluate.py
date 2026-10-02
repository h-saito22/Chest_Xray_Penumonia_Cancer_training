"""学習済みモデルの評価(accuracy/precision/recall/F1、誤分類インデックスの取得)をまとめたモジュール。"""

import numpy as np
import torch
import torch.nn as nn


## FN(またはFP)の誤分類を確認する
def calculation_test(model, dataset, epoch_num=1, recall=True):
    criterion = nn.CrossEntropyLoss()
    mean_list = []
    TP = FP = FN = TN = 0
    mat_list = []
    false_detect = []
    for epoch in range(epoch_num):
        loss_list = []
        for i, (images, labels) in enumerate(dataset):
            outputs = model(images)
            pred_label = torch.argmax(outputs, dim=1)
            if recall:
                local_idx = torch.where((pred_label == 0) & (labels == 1))[0]
            else:
                local_idx = torch.where((pred_label == 1) & (labels == 0))[0]
            global_idx = i * 16 + local_idx
            loss = criterion(outputs, labels)
            loss_value = loss.item()
            loss_list.append(loss_value)
            TP += ((pred_label == 1) & (labels == 1)).sum().item()
            FP += ((pred_label == 1) & (labels == 0)).sum().item()
            FN += ((pred_label == 0) & (labels == 1)).sum().item()
            TN += ((pred_label == 0) & (labels == 0)).sum().item()
            false_detect.append(global_idx)
        if epoch == 0:
            mat_list.append((TP, TN, FN, FP))
            conf_mat = np.array(mat_list)
            print(conf_mat)
        mean_loss = np.mean(loss_list)
        mean_list.append(mean_loss)
        print(f"epoch {epoch}: mean_loss = {mean_loss}")
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (recall + precision)
    return accuracy, precision, recall, F1, false_detect
