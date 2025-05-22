# -------------------------------------------------------------------------
# MVXNet – DAIR-V2X-I (infrastructure) inference Dockerfile
#   • CUDA 11.1.1 + cuDNN 8   • Ubuntu 20.04
#   • Python 3 (system + venv) • PyTorch 1.9.0 + cu111
#   • MMDetection3D fork (hoosh-ir)
# -------------------------------------------------------------------------
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. System deps + Python3 + OpenGL + SSH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        build-essential clang llvm libc++-dev ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
        openssh-server python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 1a. SSH server setup
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# 2. Use bash for all subsequent RUN steps
SHELL ["/bin/bash", "-lc"]

# 3. Expose CUDA
ENV CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} \
    CPATH=${CUDA_HOME}/include:${CPATH} \
    LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

# 4. Create a Python virtualenv and install deep-learning stack
ENV VENV_DIR=/opt/venv
RUN python3 -m venv $VENV_DIR \
    && source $VENV_DIR/bin/activate \
    && pip install --upgrade pip \
    && pip install --no-cache-dir \
        torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
          -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir \
        mmcv-full==1.3.14 \
          -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && pip install --no-cache-dir \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1 \
        open3d==0.11.0 \
        gdown \
    && deactivate

# Put venv's python & scripts first in PATH
ENV PATH=${VENV_DIR}/bin:${PATH}

# Activate virtualenv automatically on login
RUN echo "source ${VENV_DIR}/bin/activate" >> /root/.bashrc

# 5. Set CUDA architectures for any compiled ops
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 6. Clone & install MVXNet (MMDetection3D fork)
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/hoosh-ir/object_detection.git

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
