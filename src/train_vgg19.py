import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    # 1. 基本設定（HAKUSANのGPUを使用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データの前処理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. データの読み込み
    # ※データセットのパスは環境に合わせて調整してください
    data_dir = './data/hymenoptera_data'
    
    # データの自動取得（ディレクトリがない場合のみ）
    if not os.path.exists(data_dir):
        print("Downloading dataset...")
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        os.system(f"wget {url} -P ./data/")
        os.system(f"unzip -q ./data/hymenoptera_data.zip -d ./data/")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 4. モデルの構築 (VGG19_BN)
    print("Initializing VGG19...")
    model_ft = models.vgg19_bn(weights='IMAGENET1K_V1')
    
    # 最終層を今回のクラス数に合わせて付け替え
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 5. 学習ループ（簡易版）
    num_epochs = 10
    train_loss, val_loss = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            if phase == 'train': train_loss.append(epoch_loss)
            else: val_loss.append(epoch_loss)

    # 6. 結果の保存（plt.show()はスパコンで使えないためファイル保存）
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png') # 画像として保存
    print("Training complete. Loss curve saved as loss_curve.png")

if __name__ == '__main__':
    main()