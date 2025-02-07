import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm

# hayakawadesu.

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ハイパーパラメータの設定
num_epochs = 100
batch_size = 128
learning_rate = 0.1
save_dir = 'logs/SimCLR/cifar10'  # モデル保存ディレクトリ

# ディレクトリが存在しない場合は作成
os.makedirs(save_dir, exist_ok=True)

# データの前処理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10データセットのダウンロードとデータローダーの作成
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# ResNet18モデルのロード（事前学習なし）
model = resnet18(weights=None, num_classes=10)
model = model.to(device)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 最終エポックのモデルの保存
save_path = os.path.join(save_dir, f'supervised_resnet18_epoch{num_epochs}.pt')
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')