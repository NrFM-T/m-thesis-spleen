import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.data import DataLoader, load_decathlon_datalist, decollate_batch, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    Spacingd, ScaleIntensityRanged, ToTensord, AsDiscrete
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

root_dir = "/home/s2610100/m-thesis/data"
data_dir = os.path.join(root_dir, "Task09_Spleen")
datalist_json = os.path.join(data_dir, "dataset.json")
model_path = "/home/s2610100/m-thesis/outputs/spleen_model_unet.pth"
output_dir = "/home/s2610100/m-thesis/outputs/vis"
os.makedirs(output_dir, exist_ok=True)

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image", "label"]),
])

val_files = load_decathlon_datalist(datalist_json, True, "training")
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)

print(f"--- UNet 推論を開始します... (Device: {device}) ---")

with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        inputs = val_data["image"].to(device)
        labels = val_data["label"].to(device)
        
        val_outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

        labels_list = decollate_batch(labels)
        labels_convert = [post_label(l) for l in labels_list]
        
        outputs_list = decollate_batch(val_outputs)
        outputs_convert = [post_pred(o) for o in outputs_list]
        
        dice_metric(y_pred=outputs_convert, y=labels_convert)

        if i < 3:
            pred_plot = torch.argmax(val_outputs, dim=1).detach().cpu()[0]
            img_plot = val_data["image"][0, 0].cpu()
            label_plot = val_data["label"][0, 0].cpu()

            plt.figure(figsize=(18, 6))
            slice_idx = img_plot.shape[2] // 2
            
            plt.subplot(1, 3, 1)
            plt.title(f"Input Image (Case {i})")
            plt.imshow(img_plot[:, :, slice_idx], cmap="gray")
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Label")
            plt.imshow(label_plot[:, :, slice_idx])
            
            plt.subplot(1, 3, 3)
            plt.title("UNet Prediction")
            plt.imshow(pred_plot[:, :, slice_idx])
            
            save_name = os.path.join(output_dir, f"result_case_{i}_unet.png")
            plt.savefig(save_name)
            plt.close()

aggregate_dice = dice_metric.aggregate().item()
print("\n" + "="*30)
print(f"UNet 平均 Dice Score: {aggregate_dice:.4f}")
print("="*30)
dice_metric.reset()