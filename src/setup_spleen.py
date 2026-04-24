import os
import matplotlib.pyplot as plt
import torch

# monai.apps からは必要なものだけ
from monai.apps import download_and_extract 

# load_decathlon_datalist は monai.data に移動したのでこちらに含める
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd
)

# パスの設定
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")

# 現在地の確認
print(f"現在地: {os.getcwd()}")
print(f"データを探している場所: {data_dir}")

# 1. データの存在確認
if os.path.exists(data_dir):
    print(f"データが見つかりました：{data_dir}")
else:
    print(f"データが見つかりません：{data_dir}")
    print("手動で解凍（tar -xvf Task09_Spleen.tar）したか確認してください。")
    exit()

# 2. 前処理パイプライン
datalist_json = os.path.join(data_dir, "dataset.json")
train_files = load_decathlon_datalist(datalist_json, True, "training")

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
])

# 3. CacheDatasetによる高速化（最初の5件のみテスト）
print("前処理をキャッシュ中（初回のみ時間がかかります）...")
check_ds = CacheDataset(data=train_files[:5], transform=val_transforms, cache_rate=1.0)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = next(iter(check_loader))

# 4. 可視化による確認
# 画像とラベルを抽出 (Batch, Channel, H, W, D) -> (H, W, D)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"Preprocessed image shape: {image.shape}")

plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Spleen Image (Preprocessed)")
# 奥行きの中央スライスを表示
slice_idx = image.shape[2] // 2
plt.imshow(image[:, :, slice_idx].detach().cpu(), cmap="gray")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Spleen Label")
plt.imshow(label[:, :, slice_idx].detach().cpu())
plt.colorbar()

# スパコン環境向け：画面表示できない場合のために画像保存
output_plot = "check_preprocessing.png"
plt.savefig(output_plot)
print(f"確認用画像を保存しました: {output_plot}")

# GUIがある環境なら表示
# plt.show() 

print("SUCCESS: Data pipeline is ready and verified!")
