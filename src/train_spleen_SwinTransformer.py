import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR  # SwinUNETRに変更
from monai.losses import DiceLoss
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord
)

# 1. パス設定
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
output_dir = "/home/s2610100/m-thesis/outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. 前処理（訓練用）
# SwinUNETRのためにパッチサイズを 96x96x96 に設定（32の倍数が望ましい）
roi_size = (96, 96, 96)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=roi_size, pos=1, neg=1, num_samples=2, # メモリ節約のためサンプル数を少し減らしています
    ),
    ToTensord(keys=["image", "label"]),
])

# 3. データローダー
print("データリストを読み込み中...")
train_files = load_decathlon_datalist(datalist_json, True, "training")
# HAKUSANのメモリを活用するため cache_rate は 1.0 (全部) で設定
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2) # メモリ確保のためbatch_size=1を推奨

# 4. モデル・損失関数・最適化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SwinUNETRの定義 ---
model = SwinUNETR(
    #img_size=roi_size,          # RandCropで切り出したサイズと一致させる
    in_channels=1,
    out_channels=2,
    feature_size=48,            # 標準的なモデル幅
    use_checkpoint=True,        # 勾配チェックポイントを有効にしてGPUメモリを節約
    spatial_dims=3,             # 3D画像であることを明示
    #use_v2: 'bool' = True,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# 5. 訓練ループ
max_epochs = 20 # Transformer系は収束に時間がかかるため、少し多めを推奨
print(f"--- 訓練開始 (SwinUNETR / Device: {device}) ---")

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
        if step % 5 == 0:
            print(f"{epoch + 1}/{max_epochs} - batch {step}: loss = {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} 終了: Average Loss = {epoch_loss / len(train_loader):.4f}")

# 6. モデルの保存（ファイル名を変更してUNet版と混ざらないようにします）
save_path = os.path.join(output_dir, "spleen_model_swin.pth")
torch.save(model.state_dict(), save_path)
print(f"--- 完了 ---")
print(f"SwinUNETRモデルを保存しました: {save_path}")