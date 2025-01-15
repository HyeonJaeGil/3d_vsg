FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN mkdir -p /home/appuser/3d_vsg
COPY . /home/appuser/3d_vsg/
WORKDIR  /home/appuser/3d_vsg

# RUN git submodule update --recursive

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y \
    build-essential  \
    cmake \
    pkg-config  \
    htop  \
    gedit  \
    wget \
    git \
    unzip  \
    curl \
    vim \
    software-properties-common \
    net-tools \
    iputils-ping
    
RUN apt clean

# install python packages with pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN echo "[user]" >> /root/.gitconfig && \
    echo "	email = now9728@gmail.com" >> /root/.gitconfig && \
    echo "	name = HyeonJaeGil" >> /root/.gitconfig

RUN echo "PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]# '\n\
if [ -x /usr/bin/dircolors ]; then\n\
    test -r ~/.dircolors && eval \"\$(dircolors -b ~/.dircolors)\" || eval \"\$(dircolors -b)\"\n\
    alias ls='ls --color=auto'\n\
    alias grep='grep --color=auto'\n\
    alias fgrep='fgrep --color=auto'\n\
    alias egrep='egrep --color=auto'\n\
fi" >> /root/.bashrc

CMD ["/bin/bash"]