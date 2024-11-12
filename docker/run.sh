docker run -dit \
  --shm-size=8gb \
  --gpus all \
  --name htylab \
  --security-opt seccomp=unconfined \
  --rm \
  -p 9999:8888 \
  -v /NFS:/NFS \
  -e TORCH_HOME=/NFS/dataset/.cache/torch \
  -e KERAS_HOME=/NFS/dataset/.cache/keras \
  -e HF_HOME=/NFS/dataset/.cache/huggingface \
  htylab_v1
