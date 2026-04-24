import os
import torch
import numpy as np
import nibabel as nib
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import Dataset

# --- 1. 外部データがないため、その場でテスト用の「仮想CT画像」を作成 ---
test_dir = "./test_data"
os.makedirs(test_dir, exist_ok=True)
dummy_file = os.path.join(test_dir, "dummy_ct.nii.gz")

# 128x128x64のノイズデータをCT画像(NIfTI)として保存
dummy_img = np.random.rand(128, 128, 64).astype(np.float32)
# NIfTI形式として書き出し
nib.save(nib.Nifti1Image(dummy_img, np.eye(4)), dummy_file)
print(f"--- テスト用データを作成しました: {dummy_file} ---")

# --- 2. MONAIの読み込みパイプライン ---
test_data = [{"image": dummy_file}]
test_transforms = Compose([
    LoadImaged(keys=["image"]),              # 作成したファイルをロード
    EnsureChannelFirstd(keys=["image"]),     # (Channel, H, W, D) 形式に整える
])

# --- 3. 実行確認 ---
try:
    check_ds = Dataset(data=test_data, transform=test_transforms)
    check_data = check_ds[0]
    print("\n" + "="*30)
    print("SUCCESS: MONAI environment is ready!")
    print(f"Loaded Image Shape: {check_data['image'].shape}")
    print("="*30)
except Exception as e:
    print(f"\nERROR: MONAI test failed:\n{e}")