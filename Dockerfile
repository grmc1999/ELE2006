FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter

#UBUNTU UPDATES
RUN export DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN export DEBIAN_FRONTEND=noninteractive
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git-all 

#PIP
RUN python -m pip install --upgrade pip setuptools
RUN python -m pip install keras==2.0.9
RUN python -m pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN python -m pip install einops