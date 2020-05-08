# To be used as a base image for your project. In your project's image
# make sure you place your MuJoCo key at /root/.mujoco/

FROM ubuntu:18.04

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    wget \
    unzip \
    git \
    curl \
    # For building glfw
    cmake \
    xorg-dev \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    # 18.04 repo is old, install glfw from source instead
    # libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    # OpenAI baselines
    libopenmpi-dev \
    # virtualenv
    python3 \
    python3-pip \
    python3-tk \
    python3-virtualenv && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Build GLFW because the Ubuntu 18.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
RUN apt-get purge -y -v libglfw*
RUN wget https://github.com/glfw/glfw/releases/download/3.3/glfw-3.3.zip && \
  unzip glfw-3.3.zip && \
  rm glfw-3.3.zip && \
  cd glfw-3.3 && \
  mkdir glfw-build && \
  cd glfw-build && \
  cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. && \
  make -j"$(nproc)" && \
  make install && \
  cd ../../ && \
  rm -rf glfw

# MuJoCo 2.0 (for dm_control)
RUN mkdir -p /root/.mujoco && \
  wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip && \
  ln -s $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
  ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Copy over just setup.py first, so the Docker cache doesn't expire until
# dependencies change
#
# Files needed to run setup.py
# - README.md
# - roam_rl/__init__.py
# - setup.py
# - Makefile
COPY README.md /root/code/roam_rl/README.md
COPY roam_rl/__init__.py /root/code/roam_rl/roam_rl/__init__.py
COPY setup.py /root/code/roam_rl/setup.py
COPY Makefile /root/code/roam_rl/Makefile
WORKDIR /root/code/roam_rl

# Create virtualenv
ENV VIRTUAL_ENV=/root/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Prevent pip from complaining about available upgrades
RUN pip install --upgrade pip

# We need a MuJoCo key to install mujoco_py
# In this step only the presence of the file mjkey.txt is required, so we only
# create an empty file
RUN touch /root/.mujoco/mjkey.txt && \
  pip install mujoco_py<2.0 && \
  make default && \
  rm -r /root/.cache/pip && \
  rm /root/.mujoco/mjkey.txt

COPY . /root/code/roam_rl/

CMD /bin/bash

