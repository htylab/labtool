# 使用 Kaggle 官方的 Python Docker 映像作為基礎映像
FROM gcr.io/kaggle-images/python:v153

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 升級 pip
RUN pip install --upgrade pip

# 安裝最新版本的 PyTorch，支援 CUDA 12.4 和 sm_86 架構
RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安裝其他 Python 套件
RUN pip install \
    tensorflow==2.14.0 \
    pydicom \
    monai \
    torchio \
    albumentations \
    jupyterlab \
    numpy \
    scipy \
    matplotlib \
    opencv-python-headless \
    nibabel \
    nilearn \
    dipy \
    antspyx \
    SimpleITK \
    pyradiomics \
    tqdm

# 設定工作目錄
WORKDIR /NFS

# 開放 Jupyter Notebook 的埠
EXPOSE 8888

# 啟動 Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
