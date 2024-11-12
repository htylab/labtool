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
    pydicom==3.0.1 \
    monai==1.4.0 \
    torchio==0.20.1 \
    albumentations==1.4.17 \
    jupyterlab==4.2.5 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    matplotlib==3.7.5 \
    opencv-python-headless==4.10.0.84 \
    nibabel==5.2.1 \
    nilearn==0.10.4 \
    dipy==1.9.0 \
    antspyx==0.5.4 \
    SimpleITK==2.4.0 \
    pyradiomics==3.0.1 \
    tqdm==4.66.1


# 設定工作目錄
WORKDIR /NFS

# 開放 Jupyter Notebook 的埠
EXPOSE 8888

# 啟動 Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
