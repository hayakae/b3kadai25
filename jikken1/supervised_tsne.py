import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. モデルのロード
model_path = 'logs/SimCLR/cifar10/supervised_resnet18_epoch100.pt'

# ResNet-18の定義（必要なら独自の定義を使用）
from torchvision.models import resnet18
model = resnet18(num_classes=10)  # CIFAR-10に合わせて出力層を調整
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 2. CIFAR-10データセットの準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 特徴量抽出用に最終分類層を削除
model = nn.Sequential(*list(model.children())[:-1])

# テストデータを通して特徴量を取得
features = []
labels = []
with torch.no_grad():
    for images, target in test_loader:
        output = model(images).squeeze(-1).squeeze(-1)  # [batch_size, 512]
        features.append(output.numpy())
        labels.append(target.numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)


# 3. t-SNEの可視化
def plot_tsne(features, labels, class_names, output_path="logs/SimCLR/cifar10/supervised_tsne.png"):
    """
    t-SNE を用いて特徴量を可視化します。
    """
    tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
    reduced_features = tsne.fit_transform(features)

    # 可視化
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1],
        c=labels, cmap='tab10', alpha=0.7, s=10  # クラスごとに色を分ける
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.xticks([])
    plt.yticks([])
    plt.title("Supervised t-SNE visualization of CIFAR-10")
    plt.savefig(output_path)
    print(f"t-SNE visualization saved to {output_path}")
    plt.show()

# CIFAR-10のクラス名をリストとして定義
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

# t-SNE可視化の実行
plot_tsne(features, labels, class_names)