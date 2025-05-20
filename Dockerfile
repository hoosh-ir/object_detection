# ---------------------------------------------------------------------------
#  MVXNet – DAIR-V2X-I  (infrastructure-side) inference container
#    • CUDA 11.1 + cuDNN 8  (devel)            • Torch 1.9.0 + cu111
#    • Python 3.7 (via Miniconda)             • MMDetection3D fork (hoosh-ir)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
# --- 1. Core OS packages -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates     \
        build-essential clang llvm libc++-dev \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Miniconda ------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# --- 3. Create the ‘mvxnet’ env ---------------------------------------------
RUN conda create -y -n mvxnet python=3.7 && conda clean -afy
SHELL ["conda", "run", "-n", "mvxnet", "/bin/bash", "-c"]

# --- 4. DL stack exactly as in README ---------------------------------------
RUN pip install --no-cache-dir \
        torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
          -f https://download.pytorch.org/whl/torch_stable.html && \
    # pre-built mmcv for Torch 1.9 / CUDA 11.1
    pip install --no-cache-dir \
        mmcv-full==1.3.14 \
          -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install --no-cache-dir \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1 \
        open3d==0.11.0 \
        gdown

# --- 5. Clone the main project ----------------------------------------------
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/hoosh-ir/object_detection.git
WORKDIR /workspace/object_detection

# 5.a  (Optional) pin extra python deps listed in repo
COPY --from=0 /dev/null /dev/null   # dummy COPY so Docker keeps cache if requirements.txt absent
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt ; fi

# 5.b  build & install the fork (this compiles CUDA/​C++ ops)
RUN pip install --no-cache-dir -e .

# --- 6. pypcd utility --------------------------------------------------------
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/klintan/pypcd.git && \
    cd pypcd && python setup.py install && cd .. && rm -rf pypcd

# --- 7. Pre-trained MVXNet checkpoint ---------------------------------------
WORKDIR /workspace/object_detection
RUN mkdir -p checkpoints && \
    gdown --fuzzy https://drive.google.com/uc?id=1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ \
          -O checkpoints/mvxnet.pkl

# --- 8. Default command ------------------------------------------------------
# Change ‘demo/my_scene.pcd’ to your own file or override at `docker run`
CMD ["conda","run","--no-capture-output","-n","mvxnet", \
     "python","tools/test.py", \
     "configs/sv3d-inf/mvxnet/trainval_config.py", \
     "checkpoints/mvxnet.pkl", \
     "--show", "--show-dir", "out", \
     "--show-score-thr", "0.1"]
