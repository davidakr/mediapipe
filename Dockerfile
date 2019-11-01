# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cudagl:10.1-base

WORKDIR /mediapipe

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        snapd \
        curl \
        git \
        wget \
        unzip \
        python \
        python-pip \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        libssl-dev \
        mesa-common-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && apt-get install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools
RUN pip install future
RUN pip install cmake
RUN pip install numpy
RUN pip install opencv-python


# Install pistache
WORKDIR /
RUN git clone https://github.com/oktal/pistache.git && \
    cd pistache && \
    git checkout 80601424790228f191245e6856c8824c2b5e20b8 && \
    git submodule update --init && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles"  \
        -DCMAKE_BUILD_TYPE=Release .. && \
    make -j && \
    make install

WORKDIR /mediapipe


# Install bazel
ARG BAZEL_VERSION=0.26.1
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

COPY . /mediapipe/


# build the application and run it
RUN bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS     mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu_webserver
CMD GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu_webserver   --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile_extended.pbtxt  9090