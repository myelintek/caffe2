FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="simon.s.chuang@gmail.com"


ENV CAFFE2_ROOT=/opt/caffe2
WORKDIR $CAFFE2_ROOT

ARG CAFFE2_VERSION=master
ENV CAFFE2_VERSION=${CAFFE2_VERSION}

ENV NCCL_DEBUG=VERSION

# caffe2 install with gpu support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    ninja \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado && \
    pip install --upgrade git+https://github.com/Maratyszcza/PeachPy && \
    pip install --upgrade git+https://github.com/Maratyszcza/confu

########## INSTALLATION STEPS ###################
RUN git clone -b ${CAFFE2_VERSION} --depth 1 --recursive https://github.com/myelintek/caffe2.git . && \
    mkdir build && cd build && \
    cmake -DCUDA_ARCH_NAME=Pascal -DCUDA_ARCH_BIN="61" -DCUDA_ARCH_PTX="61" -DUSE_NNPACK=ON -DUSE_ROCKSDB=OFF -DUSE_NCCL=ON .. && \
    make -j"$(nproc)" install && \
    ldconfig && \
    make clean && \
    cd .. && \
    rm -rf build

RUN mkdir /workspace && \
    cp -rf /opt/caffe2/caffe2/python/tutorials/* /workspace/

WORKDIR /workspace
ENV PYTHONPATH /usr/local

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
