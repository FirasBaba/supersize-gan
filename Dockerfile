# Use your preferred base image
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Install necessary dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    wget \
    curl \
    ca-certificates

RUN DEBIAN_FRONTEND=noninteractive apt-get install libffi-dev

# Download and install Python 3.11
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar xzf Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/Python-3.11.0*

WORKDIR /src/
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Set the working directory

