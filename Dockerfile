FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter

#UBUNTU UPDATES
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install git-all -y

#PIP
RUN python -m pip install --upgrade pip setuptools
RUN python -m pip install keras==2.0.9
RUN python -m pip install git+https://www.github.com/keras-team/keras-contrib.git