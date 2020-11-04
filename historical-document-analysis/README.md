# What in this folder

The docker-compose.yml file in this folder will build two dockers. One is MongoDB, we will pull the official image to build. Another docker contains flask and detectron2, which are used to infer and build web pages. This docker will be built through the Dockerfile. In addition to building detectron2, this dockerfile will copy all the code in this folder to the container. And execute our main program when it start.

The DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml file is used to build the model. Note, this yaml file is not the weight of the model, but describes the model we want to use, that is, tells detectron2 how to build this model. 
In order to use our model, we need to provide a trained model weight as well . We trained a model with AP = 0.6. You can download it [here](https://drive.google.com/file/d/1-tUA7c8Mlsxwh1hiCldrs-6sDNqQivGm/view?usp=sharing). 

The requirements.txt contains the libraries needed. In the process of building docker through dockerfile, all libraries will be installed according to this file.


# How to use
To run this project, you need to install docker and docker compose. The test platform is docker based on WSL2. CUDA is not necessary. If your device has CUDA, it will be used by default. If you don't want to use CUDA, then you can modify it in code/deteinfer.py.

## Use our model weight
Copy this folder in your local machine, download the model weight from the above link, copy the model weight into code folder. Than, cd to this folder and run `docker-compose up`. When you see 2 dockers start, open the browser and enter localhost:5000. If all goes well, you should see the start page.

In addition, the detectron2 image is more than 7 GB, it may take half an hour to download if the image not available locally.

## Use your own model
You can use the weights trained by yourself, but if the model you used is not in the config folder provided by detectron2, you need to provide a corresponding YAML file as well, just like the DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml in this folder.

After that, follow this instruction:
1. Modify the penultimate command in the dockerfile, change DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml to your own yaml.
2. Copy your model weight to code folder.
3. Go to code/maincode.py modify line 18,19. Use your model and weights instead of ours.
