# -------------------------------------------------------------------------
# MVXNet – DAIR-V2X-I (infrastructure) inference Dockerfile
#   • CUDA 11.1.1 + cuDNN 8   • Ubuntu 20.04
#   • Python 3.7 (conda)      • PyTorch 1.9.0 + cu111
#   • MMDetection3D fork (hoosh-ir)
# -------------------------------------------------------------------------
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. System deps (runtime + build) + X/OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        build-essential clang llvm libc++-dev ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2. Expose CUDA in conda shells and make its headers/libs visible
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPATH=${CUDA_HOME}/include:${CPATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}

# 3. Miniconda install
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# 4. Create mvxnet env
RUN conda create -y -n mvxnet python=3.7 && conda clean -afy

# From here, every RUN is inside mvxnet
SHELL ["conda", "run", "-n", "mvxnet", "/bin/bash", "-c"]

# 5. Install the deep-learning stack
RUN pip install --no-cache-dir \
        torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
          -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir \
        mmcv-full==1.3.14 \
          -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && pip install --no-cache-dir \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1 \
        open3d==0.11.0 \
        gdown

# 6. Define CUDA architectures for extension build
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 7. Clone & install MVXNet (MMDetection3D fork)
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/hoosh-ir/object_detection.git
WORKDIR /workspace/object_detection

# 7a. Any extra Python requirements
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt ; \
    fi

# 7b. Compile & install C++/CUDA ops
RUN pip install --no-cache-dir -e .

# 8. Clone & install pypcd
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/klintan/pypcd.git \
    && cd pypcd \
    && python setup.py install \
    && cd .. \
    && rm -rf pypcd

# 9. Download the pre-trained MVXNet checkpoint
WORKDIR /workspace/object_detection
RUN mkdir -p checkpoints \
    && gdown --fuzzy https://drive.google.com/uc?id=1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ \
             -O checkpoints/mvxnet.pkl

# 10. (Optional) Remove build tools to slim the image
RUN apt-get purge -y build-essential clang llvm libc++-dev ninja-build \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 11. (Optional) Switch to non-root user
RUN useradd -m -s /bin/bash app \
    && chown -R app:app /workspace
USER app

# 12. Default command: demo inference
CMD ["conda","run","--no-capture-output","-n","mvxnet", \
     "python","tools/test.py", \
     "configs/sv3d-inf/mvxnet/trainval_config.py", \
     "checkpoints/mvxnet.pkl", \
     "--show","--show-dir","out","--show-score-thr","0.1"]
