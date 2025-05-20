# ------------------------------------------------------------
# MVXNet (DAIR-V2X-I) inference container
# CUDA 11.1 -- Torch 1.9 -- Python 3.7
# ------------------------------------------------------------
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# --- 1. OS-level packages ----------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential     \
        git                 \
        wget                \
        curl                \
        ca-certificates     \
        llvm                \
        clang               \
        libc++-dev          \
        libglib2.0-0        \
        libsm6              \
        libxrender1         \
        libxext6            \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Miniconda w/ Python 3.7 ---------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# --- 3. Create the exact env described ---------------------------------------
RUN conda create -n mvxnet python=3.7 -y && conda clean -afy
SHELL ["conda", "run", "-n", "mvxnet", "/bin/bash", "-c"]

# --- 4. Deep-learning & Open3D stack ----------------------------------------
RUN pip install --no-cache-dir \
        torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
          -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
        mmcv-full==1.3.14                 \
          -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install --no-cache-dir \
        mmdet==2.14.0                     \
        mmsegmentation==0.14.1            \
        open3d==0.11.0                    \
        gdown

# --- 5. Project code ---------------------------------------------------------
# Copy your repo (must contain object_detection/ and related configs/tools)
WORKDIR /workspace/mvxnet
COPY . .

# Editable install of mmdetection3d fork under object_detection/
RUN cd object_detection && pip install -e . && cd -

# --- 6. Extra utilities ------------------------------------------------------
RUN git clone --depth 1 https://github.com/klintan/pypcd.git && \
    cd pypcd && python setup.py install && cd .. && rm -rf pypcd

# --- 7. Checkpoints ----------------------------------------------------------
RUN mkdir -p checkpoints && \
    gdown --fuzzy https://drive.google.com/uc?id=1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ \
          -O checkpoints/mvxnet.pkl

# --- 8. Default command ------------------------------------------------------
#   -t   path to model config      (edit if your path differs)
#   -w   checkpoint .pkl           (already downloaded)
#   -i   example .pcd to test
CMD ["conda","run","--no-capture-output","-n","mvxnet", \
     "python","tools/inference.py", \
     "--config","configs/mvxnet_dair_v2x_i.py", \
     "--checkpoint","checkpoints/mvxnet.pkl", \
     "--input","demo/infra_sample.pcd"]
