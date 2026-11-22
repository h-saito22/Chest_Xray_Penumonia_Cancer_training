#各モジュールをインポートする。
import argparse
from dataset import get_dataloaders
from train import BaseLine_CNN_train , resnet_train , efficient_train
from models import ResNet_finetuning ,get_resnet_frozen , get_efficientnet_setup , get_BaseLine_CNN
from evaluate import calculation_test
from utils import model_params_count , mesuare_interface_time 
import numpy as np
import torch , random

def set_seed(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(name, model, optimizer, train_fn, loaders, epoch_num=3):
    train_data = loaders["train"]
    val_data = loaders["val"]
    test_data = loaders["test"]

    # 学習
    train_fn(model, optimizer, train_data, epoch_num,name)

    # 評価(test / val)
    test_acc, test_precision, test_recall, test_f1, test_false_detect = calculation_test(model, test_data)
    val_acc, val_precision, val_recall, val_f1, val_false_detect = calculation_test(model, val_data)

    # パラメータ数・推論時間
    params = model_params_count(model)
    # inference_time = mesuare_interface_time(model, train_data)

    print(f"=== {name} ===")
    print(f"test : acc={test_acc:.4f} precision={test_precision:.4f} recall={test_recall:.4f} f1={test_f1:.4f}")
    print(f"val  : acc={val_acc:.4f} precision={val_precision:.4f} recall={val_recall:.4f} f1={val_f1:.4f}")
    
    
    return {
        "name": name,
        "test_acc": test_acc, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1,
        "val_acc": val_acc, "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1,
        "params": params,
    }

if __name__ == "__main__":
    EXPERIMENTS = {
        "baseline": (get_BaseLine_CNN, BaseLine_CNN_train),
        "resnet_frozen": (get_resnet_frozen, resnet_train),
        "resnet_finetuning": (ResNet_finetuning, resnet_train),
        "resnet_finetuning_noise": (ResNet_finetuning, resnet_train),   # ← 追加
        "efficientnet": (get_efficientnet_setup, efficient_train),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--norm", choices=["dataset", "imagenet"], default="dataset")   # ← 追加
    args = parser.parse_args()

    set_seed(42)    # get_dataloaders の前
    use_noise = args.model == "resnet_finetuning_noise"
    train_data_loader, val_data_loader, test_data_loader = get_dataloaders(noise=use_noise, norm=args.norm)   # ← norm を渡す
    loaders = {"train": train_data_loader, "val": val_data_loader, "test": test_data_loader}

    setup_fn, train_fn = EXPERIMENTS[args.model]
    model, optimizer = setup_fn()
    run_experiment(f"{args.model}_{args.norm}", model, optimizer, train_fn, loaders)   # ← 名前に norm を付ける