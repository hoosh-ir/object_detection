FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. System deps + OpenGL + SSH server
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        build-essential clang llvm libc++-dev ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx libglu1-mesa libx11-dev \
        openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup SSH server
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
ENV CPATH=${CUDA_HOME}/include:${CPATH:-}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH:-}

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# Create conda env
RUN conda create -y -n mvxnet python=3.7 && conda clean -afy

# Install deep learning packages in one RUN to avoid shell switching issues
RUN /opt/conda/bin/conda run -n mvxnet pip install --no-cache-dir \
        torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
          -f https://download.pytorch.org/whl/torch_stable.html && \
    /opt/conda/bin/conda run -n mvxnet pip install --no-cache-dir \
        mmcv-full==1.3.14 \
          -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    /opt/conda/bin/conda run -n mvxnet pip install --no-cache-dir \
        mmdet==2.14.0 mmsegmentation==0.14.1 open3d==0.11.0 gdown

# Define CUDA architectures
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

WORKDIR /workspace

# Clone and install MVXNet
RUN git clone --depth 1 https://github.com/hoosh-ir/object_detection.git
WORKDIR /workspace/object_detection

RUN if [ -f requirements.txt ]; then \
        /opt/conda/bin/conda run -n mvxnet pip install --no-cache-dir -r requirements.txt ; \
    fi

RUN /opt/conda/bin/conda run -n mvxnet pip install --no-cache-dir -e .

WORKDIR /workspace

# Clone and install pypcd
RUN git clone --depth 1 https://github.com/klintan/pypcd.git \
    && cd pypcd \
    && /opt/conda/bin/conda run -n mvxnet python setup.py install \
    && cd .. \
    && rm -rf pypcd

WORKDIR /workspace/object_detection

RUN mkdir -p checkpoints \
    && /opt/conda/bin/conda run -n mvxnet gdown --fuzzy https://drive.google.com/uc?id=1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ \
             -O checkpoints/mvxnet.pkl

# Setup conda initialization script and shell profiles
RUN echo '#!/bin/bash\n\
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"\n\
if [ $? -eq 0 ]; then\n\
    eval "$__conda_setup"\n\
else\n\
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then\n\
        . /opt/conda/etc/profile.d/conda.sh\n\
    else\n\
        export PATH=/opt/conda/bin:$PATH\n\
    fi\n\
fi\n\
unset __conda_setup\n\
conda activate mvxnet' > /root/conda_init.sh && chmod +x /root/conda_init.sh

RUN echo 'source /root/conda_init.sh' >> /root/.bashrc && \
    echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> /root/.bash_profile

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
