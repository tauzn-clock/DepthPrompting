FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \ 
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo 123

RUN git clone --branch metric_3d https://github.com/tauzn-clock/DepthPrompting 
WORKDIR /DepthPrompting

RUN chmod +x ./requirements.sh
RUN ./requirements.sh
