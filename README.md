### 1.プロジェクト概要
医療画像[胸部X線画像](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)を分類する既存研究や公開データセットを調査し、社内PoCとして画像分類パイプラインを作る。
### 使用データセット

医療画像[胸部X線画像](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
### 実行環境
ターミナル上で以下のコマンドを実行
1. pip install -U pip 

2. pip install -r requirements.txt
### ディレクトリ構成

```text
.
├── README.md
├── notebooks/
│   └── pneumonia_classification.ipynb
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── data/
│   ├── test
│   ├── train
│   └── val
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── requirements.txt
```

※ `outputs/`(学習済みモデルや量子化実験の成果物)は`.gitignore`で除外しており、リポジトリには含まれない。`main.py`や各ノートブックを実行すると、同じ場所に再生成される。

### 実行手順
1. /src直下にあるmain.pyを実行する。
実行方法は以下の通りである。
   - BaseLineCNN: python main.py --model baseline
   - ResNet18:  python main.py --model resnet_frozen
   - ResNet18_finetuning: python main.py --model resnet_finetuning 
   - Efficient-B0: python main.py --model efficientnet 

### モデル構成
今回はBaseline CNN、ResNet18、EfficientNet-B0の3種類のモデルを実装し、比較した。さらにResNet18については、backboneを凍結したLinear Probing版に加えて、`layer4`を解凍したFineTuning版(ガウシアンノイズの有無で2パターン)も追加で検証した。

- **Baseline CNN**: 自作の軽量CNN(Conv+Pool 1セット + 全結合層)。全パラメータをゼロから学習。
- **ResNet18 (frozen)**: torchvisionのImageNet事前学習済みモデル。backbone(畳み込み層全体)を凍結し、最終層(`fc`)のみを2クラス出力の`nn.Linear`に置き換えて学習(Linear Probing)。
- **ResNet18 (FineTuning)**: 同じくImageNet事前学習済みのResNet18をベースに、backboneのうち出力層に最も近い`layer4`のみ凍結を解除し、新しい`fc`と合わせて学習(差分学習率: layer4=0.0001, fc=0.001)。過学習対策として、学習データ(Tensor化後)にガウシアンノイズを加えるAugmentationを追加したバージョンも比較している(詳細は考察を参照)。
- **EfficientNet-B0**: 同じくImageNet事前学習済みモデル。backboneを凍結し、`classifier`内の最終`Linear`層のみを2クラス出力に置き換えて学習。

共通の学習設定:
- 損失関数: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- エポック数: 3
- 入力画像: 224×224、RGB3チャンネル、ImageNet統計値で正規化(ResNet18・EfficientNet-B0)

### 評価指標
- Accuracy(正解率)、Precision(適合率)、Recall(再現率)、F1-score を使用。
- 本タスクは肺炎の見落とし(偽陰性 / FN)を避けることが特に重要な医療スクリーニングタスクであるため、Recallを重視して評価した。

### 実験結果

**test_data**

| モデル | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Baseline CNN | 0.798 | 0.764 | 0.979 | 0.858 |
| ResNet18 (frozen) | 0.598 | 0.638 | 0.823 | 0.719 |
| ResNet18 (FineTuning, ノイズなし) | 0.628 | 0.653 | 0.864 | 0.744 |
| ResNet18 (FineTuning, ノイズあり) | 0.638 | 0.651 | 0.908 | 0.758 |
| EfficientNet-B0 | 0.885 | 0.886 | 0.936 | 0.910 |

**val_data**

| モデル | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Baseline CNN | 0.75 | 0.667 | 1.0 | 0.8 |
| ResNet18 (frozen) | 0.6875 | 0.615 | 1.0 | 0.762 |
| EfficientNet-B0 | 0.9375 | 0.889 | 1.0 | 0.941 |

※ResNet18 (FineTuning)版についてはval_dataでの評価は未実施。

**モデルサイズ・推論処理時間**(train_data全体をCPUで推論した場合の処理時間)

| モデル | パラメータ数 | 推論処理時間 |
|---|---|---|
| Baseline CNN | 21,962 | 59.9s |
| ResNet18 | 11,177,538 | 332.1s |
| EfficientNet-B0 | 4,010,110 | 969.0s |

※ResNet18のFineTuning版も同一アーキテクチャのため、パラメータ数・推論処理時間はResNet18(frozen)と同じ値になる(学習時にどの層を凍結したかは推論速度やモデルサイズに影響しない)。

### 考察
- **Baseline CNN vs ResNet18**: 同じ「backboneを凍結して分類層だけ学習する」設定にも関わらず、ResNet18はBaseline CNNより全指標で劣る結果となった。trainの精度(0.90)自体は悪くなかったことから、原因は学習エポック不足よりも、ImageNet(自然画像)で学習された特徴が胸部X線画像(医療画像)のドメインにうまく転移できていない(ドメインギャップ)ことが主な要因と考えられる。
- **ResNet18 vs EfficientNet-B0**: 同じ凍結方式にも関わらず、EfficientNet-B0はBaseline CNNをも上回る結果となった。EfficientNet-B0はResNet18よりパラメータ数が少ないにも関わらずImageNetでの分類精度は高いことが知られており、backbone自体が抽出する特徴量の質の違いが、転移学習の成果に直結したと考えられる。
- **誤分類分析(FP/FN)**: いずれのモデルについても、誤分類された画像を目視で確認したが、明確な視覚的特徴(明るさなど)は見られなかった。誤分類の要因は画像自体の異常というより、モデル側(表現力やドメイン適応)にあると考えられる。
- **モデルサイズと推論速度のトレードオフ**: EfficientNet-B0はパラメータ数が最も少ないにも関わらず、CPU上での推論処理時間は3モデル中最も長かった。これは、EfficientNet-B0が多用するDepthwise Separable ConvolutionがCPU上では効率的に計算されにくいためと考えられる。パラメータ数と実際の処理速度は必ずしも比例しないことが分かった。なお、この計測はCPU環境での結果であり、GPU環境では異なる結果になる可能性がある。
- **ResNet18のFineTuningと過学習について**: 上記の通り、ResNet18(frozen)はBaseline CNNやEfficientNet-B0より精度が低かったため、出力層に最も近い`layer4`の固定を解除してFineTuningを行った(差分学習率: layer4=0.0001, fc=0.001)。その結果、test_accは0.598→0.628まで改善したが、学習済みモデルのtrain_acc(0.966)とtest_acc(0.628)の差は約33.8ポイントあり、過学習に近い振る舞いが見られた。
  過学習対策として、ピクセルをテンソルに変換した後にガウシアンノイズを加えるAugmentationを追加したところ、test_accは0.628→0.638、recallは0.864→0.908まで改善した(ノイズ単体の効果として、FineTuning同士で比較)。しかし、train_acc(0.972)とtest_acc(0.638)の差は約33.4ポイントとほぼ変化がなく、ノイズによる過学習の抑制効果は限定的だった。
  ResNet18の`layer4`はResNet18全体(約1170万パラメータ)のうち約840万パラメータ(7割以上)を占めており、今回の学習データ量(train 5000枚程度)に対してパラメータ数が多すぎることが、過学習の主なボトルネックになっていると考察する。ガウシアンノイズのような軽いAugmentationでは、このボトルネック自体は解消できなかった。
- **総合的なモデル選定**: 精度だけで見ればEfficientNet-B0が最も優れていたが、本タスクを「一次スクリーニング」として位置づけた場合、Recall(見落としの少なさ)と推論速度を優先すべきと考えた。Baseline CNNはRecallが約98%と最も高く、推論速度も最速であったため、まず広くスクリーニングし、陽性と判定された患者に対して二次的に精密検査を行うという運用を想定すると、Baseline CNNが実用上のバランスに優れると結論づけた。

### 今後の改善案
- ResNet18のFineTuning版について、weight decayやdropoutなどの正則化、解凍する層の範囲を見直し、`layer4`のパラメータ数過多による過学習ボトルネックの根本解決を検討する
- EfficientNet-B0についても、backboneを凍結せずにfine-tuning(全体または一部の層を解放)した場合の性能を検証する
- データ拡張(左右反転の有無、ガウシアンノイズ・ぼかしの追加)によるロバスト性への影響をablationで検証する
- GPU環境での推論速度を再計測し、CPU環境との違いを確認する
- val_dataのサンプル数が16枚と少なく評価が不安定になりやすいため、より大きなvalidation setでの再評価を行う
- (完了)PyTorchの量子化によるCPU推論のモデルサイズ・速度の軽量化を検証した。詳細は下記「量子化による推論最適化の検証」を参照。

### 量子化による推論最適化の検証

**目的**: 01で学習したモデルをGPUのないCPU環境で運用する場合を想定し、量子化によってモデルサイズ・CPU推論時間がどの程度改善し、Accuracy/Recall/F1-scoreにどの程度影響するかを検証した。詳細な演習内容は`02_pytorch-quantization-inference-optimization.md`を参照。

**対象モデル**: ResNet18 (frozen、Linear Probing版)

**測定環境・条件**:
- CPU推論(ローカル環境、GPU不使用)
- test_data、batch size 16、画像サイズ224×224、ImageNet統計値で正規化(01と同一条件)
- 推論時間は同条件で6回測定し、最初の3回をウォームアップとして除外、残り3回の平均値・標準偏差を記録

**試した手法**:
1. **動的量子化**(PyTorch, `torch.quantization.quantize_dynamic`, `nn.Linear`のみ対象)
2. **静的量子化(PyTorch eager mode)**: `QuantStub`/`DeQuantStub`で入出力の変換モデルまで実装したが、ResNet18の残差接続(`out += identity`)がモジュール単位の自動変換(`prepare`/`convert`)の対象外であり、量子化されたテンソル同士の加算に対応する実装が存在しないため、実行時エラーとなった。`prepare`/`convert`は`nn.Conv2d`や`nn.Linear`のようなモジュール単位でしか自動置換を行わないため、生の演算子(`+=`)には対応できないことが原因。`torchvision.models.quantization.resnet18`のような、量子化を見越して`FloatFunctional`で残差接続を実装したアーキテクチャへの置き換えが必要と判明し、今回はここで区切った。
3. **静的量子化(ONNX Runtime)**: PyTorchモデルをONNX形式にエクスポートし、ONNX Runtimeの`quantize_static`で量子化。計算グラフ全体に対して量子化を適用する方式のため、残差接続を含むモデルでも実行できた。

**比較結果**

| モデル | 量子化手法 | Accuracy | Recall | F1-score | サイズ | CPU推論時間(平均±標準偏差) |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 (frozen) | なし(PyTorch, float32) | 0.593 | 0.908 | 0.736 | 42.7MB | 30.24s ± 0.60s |
| ResNet18 (frozen) | 動的量子化(PyTorch) | 0.595 | 0.910 | 0.737 | 42.7MB(ほぼ変化なし) | 36.46s ± 2.51s(悪化) |
| ResNet18 (frozen) | なし(ONNXエクスポートのみ、比較用基準) | 0.580 | 0.815 | 0.708 | 42.6MB | 29.10s ± 0.82s |
| ResNet18 (frozen) | 静的量子化(ONNX Runtime) | 0.506 | 0.554 | 0.584 | 11.2MB(約1/4) | 11.03s ± 0.73s(約2.6倍速) |

※PyTorchとONNXの評価には同一のテストデータを使用している。量子化による性能変化を確認するため、PyTorchの量子化前後（1行目と2行目）、ONNXの量子化前後（3行目と4行目）をそれぞれ比較している。

**精度低下の有無**: 動的量子化は`nn.Linear`(ResNet18全体の約0.1%のパラメータ)のみを対象とするため、サイズ・精度ともにほぼ変化がなく、推論時間はむしろ悪化した。一方、ONNX Runtimeによる静的量子化は、畳み込み層を含むモデル全体をint8化したため、サイズ(約1/4)と速度(約2.6倍)は明確に改善したが、Recallが0.815→0.554と大幅に低下した。

**採用判断**: 動的量子化は、サイズ・速度のいずれにも改善が見られず、このモデルには適さない(CNN中心のアーキテクチャでは対象パラメータが少なすぎるため)。ONNX Runtimeの静的量子化は、サイズ・速度の両面で明確な効果があったが、本タスクは肺炎の見逃し(偽陰性)を避けることが重要な医療スクリーニングタスクであり、Recallが26ポイント以上低下する結果は許容できない。したがって、現状の設定のまま量子化モデルを採用することは推奨しない。

**今後試したい軽量化手法**:
- `torchvision.models.quantization.resnet18`のような量子化対応アーキテクチャ(`FloatFunctional`で残差接続を実装)に置き換え、PyTorch eager modeでの静的量子化を再検証する
- per-channel quantizationや、MinMaxなど異なるキャリブレーション手法を試し、Recall低下を抑えられないか検証する
- Quantization Aware Training(QAT)を用いて、量子化を前提とした再学習でRecall低下を軽減できないか検証する
- 同様の検証をEfficientNet-B0にも適用する
- TorchScriptなど、量子化以外の軽量化手法も比較対象に加える

