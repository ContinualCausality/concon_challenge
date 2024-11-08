FROM nvidia/cuda:11.4.3-base-ubuntu20.04

WORKDIR /workspace

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git 

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    kaggle \
    wandb \
    matplotlib \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    seaborn \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html \
    tensorboard \
    Pillow \
