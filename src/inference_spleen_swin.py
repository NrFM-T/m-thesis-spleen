import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, load_decathlon_datalist, decollate_batch, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord, AsDiscrete
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- パス設定 ---
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
# 学習で得られた「最高精度」の重みを指定
model_path = "/home/s2610100/m-thesis/outputs/best_metric_model.pth"
output_dir = "/home/s2610100/m-thesis/outputs/vis"
os.makedirs(output_dir, exist_ok=True)

# --- 前処理 (学習時の検証用Transformsと完全に一致させる) ---
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"], 
        pixdim=(1.5, 1.5, 2.0), 
        mode=("bilinear", "nearest")
    ),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-57, a_max=164, 
        b_min=0.0, b_max=1.0, 
        clip=True
    ),
    ToTensord(keys=["image", "label"]),
])

# --- データの読み込み (重要：学習に使っていないデータのみを抽出) ---
# MSD形式のjsonから全リストを取得
all_files = load_decathlon_datalist(datalist_json, True, "training")

# 学習時と同じ分割ルールを適用（例：後ろから9件を検証用としていた場合）
# ※学習コードで random_split を使用していた場合は、シード値を固定して再現する必要があります
val_files = all_files[-9:] 

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

# --- モデル構築と重みのロード ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    in_channels=1, 
    out_channels=2, 
    feature_size=48,
    use_checkpoint=True, 
    spatial_dims=3,
).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Weights loaded: {model_path}")
else:
    print(f"Error: Model not found at {model_path}")
    exit()

model.eval()

# --- 評価指標の設定 ---
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

print(f"\n--- SwinUNETR 推論を開始します... (検証データ数: {len(val_files)}) ---")

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (96, 96, 96) 
        sw_batch_size = 4
        inputs = val_data["image"].to(device)
        labels = val_data["label"].to(device)
        
        # スライディングウィンドウ推論の実行
        val_outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

        # 後処理とDiceスコア計算
        labels_list = decollate_batch(labels)
        labels_convert = [post_label(l) for l in labels_list]
        
        outputs_list = decollate_batch(val_outputs)
        outputs_convert = [post_pred(o) for o in outputs_list]
        
        dice_metric(y_pred=outputs_convert, y=labels_convert)

        # 可視化 (Case 0-2 を保存)
        if i < 3:
            # 予測結果のチャンネル0(背景)と1(脾臓)から、値が大きい方を採用して0/1画像にする
            pred_plot = torch.argmax(val_outputs, dim=1).detach().cpu()[0]
            img_plot = val_data["image"][0, 0].cpu()
            label_plot = val_data["label"][0, 0].cpu()

            plt.figure(figsize=(18, 6))
            slice_idx = img_plot.shape[2] // 2 # 奥行きの中心を表示
            
            plt.subplot(1, 3, 1)
            plt.title(f"Input Image (Case {i})")
            plt.imshow(img_plot[:, :, slice_idx], cmap="gray")
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth (Label)")
            plt.imshow(label_plot[:, :, slice_idx])
            
            plt.subplot(1, 3, 3)
            plt.title("SwinUNETR Prediction")
            plt.imshow(pred_plot[:, :, slice_idx])
            
            save_name = os.path.join(output_dir, f"result_case_{i}_swin_fix.png")
            plt.savefig(save_name)
            plt.close()
            print(f"Saved visualization: {save_name}")

# 最終的な平均スコアを表示
aggregate_dice = dice_metric.aggregate().item()
print("\n" + "="*40)
print(f"SwinUNETR 検証データ平均 Dice Score: {aggregate_dice:.4f}")
print("="*40)
dice_metric.reset()