import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from models import SimCLR  # SimCLR モデルをインポート
import torchvision.models as models  # ベースエンコーダとして使用

import warnings
warnings.filterwarnings("ignore")

def load_test_data(data_dir, batch_size):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # データの正規化
    ])
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


def extract_features(model, data_loader, device):
    """
    モデルを使ってテストデータの特徴ベクトルを抽出します。
    """
    model.eval()  # モデルを評価モードに設定
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            lbls = lbls.numpy()  # クラスラベルをnumpy配列に変換
            feature, _ = model(images)  # 特徴ベクトルを取得
            features.append(feature.cpu().numpy())
            labels.append(lbls)
    return np.concatenate(features), np.concatenate(labels)


def plot_tsne(features, labels, class_names, output_path="logs/SimCLR/cifar10/simclr_tsne.png"):
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
    plt.title("SimCLR t-SNE visualization of CIFAR-10")
    plt.savefig(output_path)
    print(f"t-SNE visualization saved to {output_path}")
    plt.show()


def main():
    # パラメータ設定
    data_dir = 'data'  
    model_path = 'logs/SimCLR/cifar10/simclr_resnet18_epoch100.pt'
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]  # CIFAR-10のクラス名

    # テストデータをロード
    test_loader = load_test_data(data_dir, batch_size)

    # モデルを初期化して読み込み
    base_encoder = models.resnet18  # 使用したバックボーンを指定
    projection_dim = 128  # SimCLRで使用した次元
    model = SimCLR(base_encoder, projection_dim).to(device)  # モデルをデバイスに移動
    model.load_state_dict(torch.load(model_path))  # 保存されたモデルを読み込み

    # 特徴ベクトルを抽出
    features, labels = extract_features(model, test_loader, device)

    # t-SNE をプロット
    plot_tsne(features, labels, class_names)


if __name__ == '__main__':
    main()
