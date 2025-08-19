# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libc++-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python

# Install pip for python3.7
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support (specific versions for compatibility)
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMDetection dependencies
RUN pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install additional ML dependencies
RUN pip install open3d==0.11
RUN pip install scikit-image
RUN pip install tensorboard
RUN pip install trimesh==2.35.39
RUN pip install networkx==2.2
RUN pip install numba==0.48.0
RUN pip install "numpy<1.20.0"
RUN pip install plyfile

# Install pypcd for point cloud loading
RUN git clone https://github.com/klintan/pypcd.git /tmp/pypcd && \
    cd /tmp/pypcd && \
    python setup.py install && \
    rm -rf /tmp/pypcd

# Install FastAPI and related dependencies
RUN pip install fastapi uvicorn

# Copy project files
COPY . /app/

# Install the project in development mode
RUN pip install -e . --user

# Create necessary directories
RUN mkdir -p /tmp /app/results

# Download pre-trained models
RUN bash scripts/download_checkpoints.sh

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["python", "app.py"]
