FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG gid
ARG uid
ARG username

ARG ROS_PACKAGES
RUN echo "$ROS_PACKAGES" > /tmp/ros_packages.sh && chmod +x /tmp/ros_packages.sh

ENV DEBIAN_FRONTEND noninteractive
ENV ROS_DISTRO noetic

# basic dependencies
RUN apt-get update && apt-get install -y \
  git \
  wget \
  sudo \
  vim \
  tmux tmuxinator \
  python3-dev \
  python3-tk \
  python3-venv \
  python3-opencv \
  lsb-release \
  gnupg2 \
  ffmpeg \
  libsuitesparse-dev \
  iputils-ping \
  libglvnd0 libgl1 libglx0 libegl1 \
  libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev \
  libxext6 libx11-6 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# for ros install
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# ros setup, for data extraction
RUN apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-desktop \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /tmp/ros_packages.sh $ROS_DISTRO

# Replace 1001 with your user / group id
ENV UID=$uid
ENV GID=$gid
ENV USERNAME=$username
RUN mkdir -p /home/${USERNAME} && \
    echo "${USERNAME}:x:${UID}:${GID}:${USERNAME},,,:/${USERNAME}:/bin/bash" >> /etc/passwd && \
    echo "${USERNAME}:x:${GID}:" >> /etc/group && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    chown ${UID}:${GID} -R /home/${USERNAME}

USER $username
ENV HOME /home/$username
WORKDIR /home/$username

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && rm get-pip.py

# https://github.com/PyTorchLightning/pytorch-lightning/issues/13159#issuecomment-1138535267=
RUN pip3 install "protobuf<4.21.0"

RUN pip3 install torch==1.13.1+cu117 \
                 torchvision==0.14.1+cu117 \
                 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install pyg_lib \
                 torch_scatter \
                 torch_sparse \
                 torch_cluster \
                 torch_spline_conv \
                 torch_geometric_temporal \
                 torch_geometric==2.3.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

#RUN pip3 install theseus-ai==0.1.4

RUN pip3 install tensorboard \
                 pandas \
                 jupyterlab==4.0.4 \
                 matplotlib==3.7.1 \
                 seaborn
# don't upgrade numpy
#  see: https://github.com/pytorch/pytorch/issues/91516
RUN pip3 install numpy==1.23.5 --upgrade
RUN pip3 install pytorch-metric-learning
RUN pip3 install torchinfo
RUN pip3 install POT
RUN pip3 install pytest
RUN pip3 install plotly
RUN pip3 install bertopic
RUN pip3 install tables
RUN pip3 install pqdm

# BLIP
RUN pip3 install timm==0.4.12 transformers==4.25.1 fairscale==0.4.4

# Stable Diffusion
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion
RUN pip3 install labml pytorch-lightning

# Sentence-Transformers
RUN pip3 install sentence-transformers

RUN pip3 install torchinfo
RUN pip3 install pipdeptree

RUN pip3 install slack_sdk

RUN pip3 install ray[default,tune]
RUN pip3 install GPUtil
RUN pip3 install captum

RUN pip3 install ipywidgets widgetsnbextension pandas-profiling


RUN echo "set -g mouse on" >> $HOME/.tmux.conf

RUN echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bash_profile
RUN echo "source $HOME/sim_ws/devel/setup.bash" >> $HOME/.bash_profile
RUN echo "cd $HOME/navigation_foundation" >> $HOME/.bash_profile
