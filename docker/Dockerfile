FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# システムパッケージと Python のインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python コマンドのシンボリックリンクを作成
RUN ln -s /usr/bin/python3 /usr/bin/python

# 作業ディレクトリの設定
WORKDIR /workspace

# パッケージリストのコピーとインストール（キャッシュ活用）
COPY docker/requirements.txt /workspace/requirements.txt

# pipのアップグレードと、requirements.txtに基づくインストール
RUN pip install --upgrade pip && \
    pip install -r requirements.txt