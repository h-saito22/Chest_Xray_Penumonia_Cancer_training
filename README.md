### 1.プロジェクト概要
医療画像[胸部X線画像](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)を分類する既存研究や公開データセットを調査し、社内PoCとして画像分類パイプラインを作る。
### 使用データセット

医療画像[胸部X線画像](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
### 実行環境
ターミナル上で以下のコマンドを実行
1. pip install -U pip 

2. pip install -r requirments.txt
### ディレクトリ構成

```text
.
├── README.md
├── notebooks/
│   └── pneumonia_classification.ipynb
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│──data/
│  ├── test
│  ├── train
│  └── val
├── outputs/
│   ├── figures/
│   └── metrics/
└── requirements.txt


```

### 実行手順

### モデル構成
### 評価指標
### 実験結果
### 考察
### 今後の改善案
### 医療利用を目的としない注意書き