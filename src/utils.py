import time
import torch    

def mesuare_interface_time(model , data):
    start_time = time.time()
    for images , labels in data :
        with torch.no_grad():
            outputs = model(images)
    end_time = time.time()
    print(f"処理時間 :{end_time - start_time}")
    return end_time - start_time

def model_params_count(model):
    count_p = 0
    for p in model.parameters():
        count_p += p.numel()
    print(count_p)
    return count_p

if __name__ == "__main__":
    from models import get_resnet_frozen , BaselineCNN 

    # 学習はせず、各モデルの構造(出力shape)だけ軽く確認する。
    dummy_input = torch.randn(1, 3, 224, 224)

    baseline_model = BaselineCNN()
    model_params_count(baseline_model)