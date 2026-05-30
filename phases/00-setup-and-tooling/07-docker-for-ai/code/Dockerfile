FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install repository utilities and build essentials, then pull Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default 'python' command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Bootstrap an up-to-date pip directly for Python 3.12 (bypasses the distutils bug)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Upgrade core packaging tools
RUN python -m pip install --no-cache-dir --upgrade setuptools wheel

# Continue with the rest of the file...
RUN python -m pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors \
    qdrant-client

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]
