import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# MONAIの最新バージョンに対応したインポート 
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,  # AddChanneldから変更 
    ScaleIntensityRanged, 
    Spacingd, 
    Orientationd, 
    ToTensord
)
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, DataLoader

# 1. 環境設定とデバイスの準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/home/s2610100/m-thesis"
model_path = os.path.join(root_dir, "outputs/best_metric_model.pth")
output_vis_dir = os.path.join(root_dir, "outputs/vis")
os.makedirs(output_vis_dir, exist_ok=True)

# 2. 推論・解析用Transformsの定義 
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # 修正ポイント 
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    ToTensord(keys=["image", "label"]),
])

# 3. 解析対象データの指定（例：case_1） 
data_list = [
    {
        "image": os.path.join(root_dir, "data/Task09_Spleen/imagesTr/spleen_2.nii.gz"),
        "label": os.path.join(root_dir, "data/Task09_Spleen/labelsTr/spleen_2.nii.gz")
    }
]
check_ds = Dataset(data=data_list, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
data_batch = next(iter(check_loader))
input_tensor = data_batch["image"].to(device)

# 4. 学習済みモデルのロード (SwinUNETR) 
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 5. Integrated Gradients (IG) の実行 
# 目的：モデルが「脾臓（クラス1）」と判定した根拠を可視化する
def model_forward(input_data):
    output = model(input_data)
    return output

ig = IntegratedGradients(model_forward)

# 入力テンソルに対して勾配が必要なことを明示
input_tensor.requires_grad = True

# 属性（アトリビューション）の算出。target=1は脾臓クラスを指す 
attribution = ig.attribute(input_tensor, target=1, n_steps=50)
attribution = attribution.detach().cpu().numpy()

# 6. 結果の可視化と保存 
img_numpy = input_tensor.detach().cpu().numpy()[0, 0]
attr_numpy = attribution[0, 0]

# スライスの中央付近を表示
slice_idx = img_numpy.shape[2] // 2

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original CT Slice")
plt.imshow(img_numpy[:, :, slice_idx], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("IG Attribution (Focus on Spleen)")
# 寄与度をヒートマップとして重ねる
plt.imshow(img_numpy[:, :, slice_idx], cmap="gray")
plt.imshow(np.abs(attr_numpy[:, :, slice_idx]), cmap="hot", alpha=0.5)
plt.colorbar()

output_file = os.path.join(output_vis_dir, "interpret_case_1.png")
plt.savefig(output_file)
print(f"SUCCESS: Interpretation result saved to {output_file}")