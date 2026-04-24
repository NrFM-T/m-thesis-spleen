import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
# load_decathlon_datalist は monai.data からインポート
from monai.data import CacheDataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord
)

# 1. パス設定（絶対パス）
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
output_dir = "/home/s2610100/m-thesis/outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. 前処理（訓練用）
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    # 脾臓がある場所を中心にパッチを切り出す
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,
    ),
    ToTensord(keys=["image", "label"]),
])

# 3. データローダー
print("データリストを読み込み中...")
train_files = load_decathlon_datalist(datalist_json, True, "training")
# CacheDatasetで高速化
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

# 4. モデル・損失関数・最適化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2, # 背景(0) と 脾臓(1)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# 5. 訓練ループ
max_epochs = 10
print(f"--- 訓練開始 (Device: {device}) ---")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if step % 2 == 0:
            print(f"{epoch + 1}/{max_epochs} - batch {step}: loss = {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} 終了: Average Loss = {epoch_loss / len(train_loader):.4f}")

# 6. モデルの保存
save_path = os.path.join(output_dir, "spleen_model.pth")
torch.save(model.state_dict(), save_path)
print(f"--- 完了 ---")
print(f"学習済みモデルを保存しました: {save_path}")