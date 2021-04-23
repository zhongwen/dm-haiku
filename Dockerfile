# FROM tensorflow/tensorflow:2.4.1-gpu
# FROM harbor.seacloud.garenanow.com/sail/cuda_python3:latest
#FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
FROM tensorflow/tensorflow
ARG DEBIAN_FRONTEND=noninteractive
RUN ln -sf /user/local/cuda-11.2 /usr/local/cuda

# Install dependencies
RUN apt-get update && apt-get install -y python3.8 python3-pip tmux ffmpeg libsm6 \
    libxext6 libxrender-dev git vim build-essential g++ cmake zlib1g-dev


# Install JAX
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade jax jaxlib==0.1.64+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install Haiku
RUN pip3 install absl-py numpy tabulate \
    dm-env>=1.2 \
    dm-tree>=0.1.1 \
    packaging>=20.9 \
    tensorflow-datasets>=4.2.0 \
    tensorflow_probability==0.11.0 \
    tensorflow \
    ml_collections \
    dm-haiku \
    rlax \
    bsuite \
    optax \
    dm-launchpad[reverb] \
    atari-py \
    gym[atari]
#RUN pip3 install git+https://github.com/deepmind/dm-haiku
#RUN pip3 install git+git://github.com/deepmind/bsuite.git
#RUN pip3 install git+git://github.com/deepmind/optax.git
#RUN pip3 install git+git://github.com/deepmind/rlax.git

# Install Atari environment
#RUN pip3 install atari-py
#RUN pip3 install gym[atari]
#RUN pip3 install dm-launchpad
RUN apt-get install tmux
ADD . /impala/
WORKDIR /impala

