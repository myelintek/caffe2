FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="simon.s.chuang@gmail.com"

ENV CAFFE2 ${CAFFE2:-master}

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
    tornado

########## INSTALLATION STEPS ###################
# RUN git clone --branch v0.8.1 --recursive https://github.com/caffe2/caffe2.git
RUN git clone --branch $CAFFE2 --recursive https://github.com/myelintek/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Pascal \
    -DCUDA_ARCH_BIN="61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=ON \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
