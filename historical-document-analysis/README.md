# What in this folder
The code folder contains our main function.  

The docker-compose.yml file in this folder will build two dockers. One is MongoDB, we will pull the official image to build. Another docker contains flask and detectron2, which are used to infer and build web pages. This docker will be built through the Dockerfile. In addition to building detectron2, this dockerfile will copy all the code in this folder to the container. And execute our maincode.py when it start.

The DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml file is used to build the model. Note, this yaml file is not the weight of the model, but describes the model we want to use, that is, tells detectron2 how to build this model. 
In order to use our model, we need to provide a trained model weight as well . We trained a model with AP = 0.6. You can download it [here](https://www.dropbox.com/s/hfhsdpvg7jesd4g/pub_model_final.pth?dl=0), if you use our build.sh, it will download automatically.

The requirements.txt contains the libraries needed. In the process of building docker through dockerfile, all libraries will be installed according to this file.


