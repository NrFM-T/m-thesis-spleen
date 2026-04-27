import os
import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord, AsDiscrete
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# --- 設定 ---
root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
output_dir = "/home/s2610100/m-thesis/outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データ準備 (Train & Val) ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=2,
    ),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

# 1. まず全データを "training" から読み込む
all_files = load_decathlon_datalist(datalist_json, True, "training")

# 2. データを分割する (例: 80%を学習、20%を検証)
# 固定のシード値(random_state)を使うことで、実行のたびに分割が変わるのを防ぎます
import random
random.seed(42)
random.shuffle(all_files)

num_val = int(len(all_files) * 0.2) # 全体の20%を検証用にする
train_files = all_files[:-num_val]
val_files = all_files[-num_val:]

print(f"Total files: {len(all_files)}")
print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

# --- 2. モデル・損失関数・評価指標 ---
model = SwinUNETR(
    in_channels=1, out_channels=2, feature_size=48,
    use_checkpoint=True, spatial_dims=3,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # AdamW + L2正則化

# 学習率スケジューラ (Cosine Annealing)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

# --- 3. 訓練ループ (バリデーション & 早期終了付き) ---
max_epochs = 300 # 長めに設定
val_interval = 2  # 2エポックごとに評価
best_metric = -1
best_metric_epoch = -1
patience = 20     # 20回連続でスコアが改善しなければ終了 (Early Stopping)
patience_counter = 0

print(f"--- SwinUNETR 訓練開始 (Device: {device}) ---")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    lr_scheduler.step()
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {epoch_loss:.4f}")

    # --- バリデーションステップ ---
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                
                # 後処理とDiceスコア計算
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            print(f"--- Validation Dice: {metric:.4f} ---")

            # ベストモデルの保存
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                patience_counter = 0 # 改善したのでリセット
                torch.save(model.state_dict(), os.path.join(output_dir, "best_metric_model.pth"))
                print(">>> SAVED NEW BEST MODEL")
            else:
                patience_counter += 1
                print(f">>> No improvement for {patience_counter} evaluation(s)")

            # 早期終了の判定
            if patience_counter >= patience:
                print(f"Early stopping triggered. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
                break

print(f"訓練完了! Best Dice: {best_metric:.4f} at Epoch: {best_metric_epoch}")