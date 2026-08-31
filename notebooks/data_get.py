# kaggle上の画像データをローカル上に保存する
import kagglehub
# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia",
                                output_dir = "./data")
print(path)