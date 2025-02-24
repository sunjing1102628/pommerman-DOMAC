FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# pay attention ARG "cuda_ver" should match base image above
ARG cuda_ver=cu111

# python 3.8.1
ARG miniconda_ver=Miniconda3-py38_4.8.2-Linux-x86_64.sh

ARG project=domac
ARG username=jsun
ARG password=jsun
ARG torch_ver=1.10.1


# Install some basic utilities and create users
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    # Create a user with home dir /home/${project}
    && useradd -md /home/${project} ${username} \
    # user owns the home dir
    && chown -R ${username} /home/${project} \
    # set user password
    && echo ${username}:${password} | chpasswd \
    # add user to sudoers
    && echo ${username}" ALL=(ALL:ALL) ALL" > /etc/sudoers.d/90-user
# switch to user
USER ${username}
# to home dir
WORKDIR /home/${project}

# download conda installer and save as "~/miniconda.sh"
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/${miniconda_ver} \
    # user owns installer
    && chmod +x ~/miniconda.sh \
    # install conda with name ~/${project}-miniconda-environment;
    # "-p" = path of installed conda env;
    # sublime open ${miniconda_ver} to check meaning of -b -p.
    && bash ~/miniconda.sh -b -p ~/${project}-miniconda-environment \
    && rm ~/miniconda.sh
ENV CONDA_AUTO_UPDATE_CONDA=false \
    # add conda to env variables
    PATH=~/${project}-miniconda-environment/bin:$PATH
RUN ~/${project}-miniconda-environment/bin/pip install torch==${torch_ver} -f https://download.pytorch.org/whl/${cuda_ver}/torch_stable.html \
    && git clone https://github.com/MultiAgentLearning/playground ~/playground \
    && cd ~/playground \
    && ~/${project}-miniconda-environment/bin/pip install -U . \
    && ~/${project}-miniconda-environment/bin/pip install opencv-python==4.5.5.62





