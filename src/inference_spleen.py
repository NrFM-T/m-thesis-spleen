import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.data import DataLoader, load_decathlon_datalist, decollate_batch, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord
)
from monai.inferers import sliding_window_inference

# 1. パス設定
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
model_path = "/home/s2610100/m-thesis/outputs/spleen_model.pth"
output_dir = "/home/s2610100/m-thesis/outputs/vis"
os.makedirs(output_dir, exist_ok=True)

# 2. 推論用前処理 (学習時と同じSpacing, Orientation)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

# 3. データ準備 (validationセットを使用)
val_files = load_decathlon_datalist(datalist_json, True, "training")
val_ds = Dataset(data=val_files, transform=val_transforms) # Cacheなしで軽量に
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

# 4. モデルの準備とロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
model.eval()

# 5. 推論と可視化
print(f"推論を開始します... (Device: {device})")
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        if i >= 3: break  # 最初の3件だけ可視化
        
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        # スライディングウィンドウ推論（大きな画像も分割して推論）
        val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
        
        # 結果を確率に変換してクラス決定 (0:背景, 1:脾臓)
        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()[0]
        img = val_data["image"][0, 0].cpu()
        label = val_data["label"][0, 0].cpu()

        # 可視化：中央のスライスを表示
        plt.figure(figsize=(18, 6))
        slice_idx = img.shape[2] // 2
        
        plt.subplot(1, 3, 1)
        plt.title(f"Input Image (Case {i})")
        plt.imshow(img[:, :, slice_idx], cmap="gray")
        
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Label")
        plt.imshow(label[:, :, slice_idx])
        
        plt.subplot(1, 3, 3)
        plt.title("Model Prediction")
        plt.imshow(val_outputs[:, :, slice_idx])
        
        save_name = os.path.join(output_dir, f"result_case_{i}.png")
        plt.savefig(save_name)
        print(f"Saved: {save_name}")
        plt.close()

print("すべて完了しました。")