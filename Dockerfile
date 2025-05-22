FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install prerequisites for adding PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.7 python3.7-venv python3.7-dev python3.7-distutils \
        python3-pip wget curl git ca-certificates \
        build-essential clang llvm libc++-dev ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Use python3.7 explicitly
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 \
    && update-alternatives --set python3 /usr/bin/python3.7

# Upgrade pip for python3.7 explicitly (use legacy script for Python 3.7)
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py

# Create virtualenv with python3.7 explicitly
ENV VENV_DIR=/opt/venv
RUN python3.7 -m venv $VENV_DIR && \
    $VENV_DIR/bin/pip install --upgrade pip && \
    $VENV_DIR/bin/pip install --no-cache-dir \
      torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
      -f https://download.pytorch.org/whl/torch_stable.html && \
    $VENV_DIR/bin/pip install --no-cache-dir \
      mmcv-full==1.3.14 \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    $VENV_DIR/bin/pip install --no-cache-dir \
      mmdet==2.14.0 \
      mmsegmentation==0.14.1 \
      open3d==0.11.0 \
      gdown


# Put venv's python & scripts first in PATH
ENV PATH=${VENV_DIR}/bin:${PATH}

# Activate virtualenv automatically on login
RUN echo "source ${VENV_DIR}/bin/activate" >> /root/.bashrc

# 5. Set CUDA architectures for any compiled ops
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 6. Clone & install MVXNet (MMDetection3D fork)
WORKDIR /workspace
RUN git clone https://github.com/hoosh-ir/object_detection.git

WORKDIR /workspace/object_detection
RUN source $VENV_DIR/bin/activate \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi \
    && pip install --no-cache-dir -e . \
    && deactivate

# 7. Clone & install pypcd
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/klintan/pypcd.git \
    && cd pypcd \
    && source $VENV_DIR/bin/activate \
    && python setup.py install \
    && deactivate \
    && cd .. \
    && rm -rf pypcd

# 8. Download the pre-trained MVXNet checkpoint
WORKDIR /workspace/object_detection
RUN mkdir -p checkpoints \
    && gdown --fuzzy https://drive.google.com/uc?id=1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ \
             -O checkpoints/mvxnet.pkl

# 9. Expose SSH port & default to running SSHD
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
