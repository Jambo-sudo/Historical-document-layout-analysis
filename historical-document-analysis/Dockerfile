# see: https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile

FROM nvidia/cuda:10.1-cudnn7-devel
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*

# 创建一个连接到Python3，防止已经安装了Python2。
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard
RUN pip install --user torch==1.6 torchvision==0.7 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# install packages from requirements 
COPY requirements.txt /home/appuser/detectron2_repo
RUN pip install --user -r /home/appuser/detectron2_repo/requirements.txt

############## 这句不知道是否必要，先把49,57注释了，然后requirements里面pillow这里的>也去掉了。如果有问题再改回来
# RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

############# 这句不知道干啥 https://spiyer99.github.io/Detectron2-Web-App/
#ENV PILLOW_VERSION=7.0.0

# copy our code to work dir，weight should in code dir
# https://drive.google.com/file/d/1-tUA7c8Mlsxwh1hiCldrs-6sDNqQivGm/view?usp=sharing
# https://www.dropbox.com/s/hfhsdpvg7jesd4g/pub_model_final.pth?dl=0
RUN wget -P /home/appuser/detectron2_repo/code https://www.dropbox.com/s/hfhsdpvg7jesd4g/pub_model_final.pth?dl=0 -O pub_model_final.pth
COPY --chown=1000:1000 code /home/appuser/detectron2_repo/code
COPY DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml /home/appuser/detectron2_repo/configs

# Make port 5000 available to the world outside the container
# ENV PORT 5000
# EXPOSE 5000

# CMD python app.py
CMD ["python3", "code/maincode.py"] 

