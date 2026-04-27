import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric  # 追加
# load_decathlon_datalist は monai.data からインポート
from monai.data import CacheDataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord,
    AsDiscrete  # 追加
)

# 1. パス設定（絶対パス）
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
output_dir = "/home/s2610100/m-thesis/outputs"
os.makedirs(output_dir, exist_ok=True)

# --- パスの存在確認（エラー対策） ---
if not os.path.exists(datalist_json):
    print(f"Error: {datalist_json} が見つかりません。")
    print("以下のコマンドで場所を確認してください: find /home/s2610100/m-thesis/data -name 'dataset.json'")
    exit()

# 2. 前処理（訓練用）
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,
    ),
    ToTensord(keys=["image", "label"]),
])

# 3. データローダー
print("データリストを読み込み中...")
train_files = load_decathlon_datalist(datalist_json, True, "training")
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

# 4. モデル・損失関数・評価指標
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2, 
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# --- 評価指標の設定 ---
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)

# 5. 訓練ループ
max_epochs = 10
best_metric = -1
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
        
        # --- Diceスコアの計算 ---
        outputs_list = [post_pred(i) for i in decollate_batch(outputs)]
        labels_list = [post_label(i) for i in decollate_batch(labels)]
        dice_metric(y_pred=outputs_list, y=labels_list)

    # エポック終了時の集計
    avg_loss = epoch_loss / len(train_loader)
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    
    print(f"Epoch {epoch + 1}/{max_epochs} - Loss: {avg_loss:.4f}, Dice: {metric:.4f}")

    # ベストスコアならモデルを保存
    if metric > best_metric:
        best_metric = metric
        save_path = os.path.join(output_dir, "best_spleen_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  --> Best model saved with Dice: {best_metric:.4f}")

print(f"--- 完了 (Best Dice: {best_metric:.4f}) ---")