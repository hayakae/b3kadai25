# B3課題20205 対照学習

## 環境構築
Dockerを用いた環境構築を行います．

1. コードのダウンロード

このフォルダを適当な場所にダウンロードします．`cd`コマンドで落としたい場所に移動後，以下のコマンドを実行してください．
```
git clone https://github.com/hayakae/b3kadai25.git
```

2. Dockerhubからベースのイメージを取得

以下のコマンドで，cuda12.4verのpytorchのイメージをダウンロードします．（cudaのverが違うかもしれないので， `nvidia-smi`でverを確認するとよいかも．）
```
sudo docker pull pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

3. 構築したい環境のイメージを作成（Dockerfileを作成）
   
適当なディレクトリに/CustomImage/Dockerfileというテキストファイルを作成します．ファイルに以下の内容をコピペしましょう．
```
# CUDA 12.4対応のベースイメージを使用
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel


# os上での日本語のエラー防止
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# 必要なシステムパッケージとPythonパッケージのインストール
RUN apt-get update && apt-get install -y \
    python-is-python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tqdm hydra-core==0.11.3 scikit-learn matplotlib ftfy regex numpy && \
    pip install --no-cache-dir \
    git+https://github.com/openai/CLIP.git

```

4. イメージのビルド

hayakawa/b3kadaというイメージが作成されます．名前とDockerファイルのパスは適宜変更してください．
```
sudo docker build  -t hayakawa/b3kadai -f /home/user/code/CustomImage/Dockerfile /home/user/code/CustomImage
```

5. イメージからコンテナを起動

イメージから，実際の動作環境であるコンテナを起動します．HostD, ContainerDのdirはb3kadaiのdirに合わせて，最終行は先ほどビルドしたイメージの名前に合わせて変更してください．
```
HostD=/home/user/code/b3kadai25 && \
ContainerD=/home/user/code/b3kadai25 && \
sudo docker run --gpus all -it \
-v "${HostD}":"${ContainerD}" \
-w "${ContainerD}" \
--shm-size=16g \
--name b3kadai \
hayakawa/b3kadai bash

```

## コードの実行

### 実験1
教師あり学習，対照学習(SimCLR)それぞれで学習したモデルの画像特徴量（Cifar-10）をt-SNEで可視化する実験です．

1. 教師あり学習

両者ともエポック数は100で設定しました．自由に変えてもいいけどこのぐらいが適当かも．
```
python jikken1/supervised.py
```
回し終わったらモデルが保存されているので可視化を行います．
```
python jikken1/supervised_tsne.py
```

1. SimCLR
```
python jikken1/simclr.py
```
可視化
```
python jikken1/simclr_tsne.py
```

