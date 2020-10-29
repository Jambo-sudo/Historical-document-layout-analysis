# Historical document layout analysis

## Background
The purpose of this project is to build a document layout analysis model and interact with users through a webpage. Since this project mainly serves a historical literature research group, this model may not be suitable for other types of input images. But we can easily deal with other tasks by using different models.

## Architecture
This project is mainly composed of two dockers. One docker including the model inference, a flask framework to build the website and accept input data. This docker also mounts a volume, which contains all the inference results image. This volume can be bound to the local storage to achieve data persistence. The model inference part uses [Mask-RCNN](https://arxiv.org/abs/1703.06870) and runs on the [detectron2](https://github.com/facebookresearch/detectron2) framework. 

The following figure shows a example of result image. The model marked out multiple layout divisions based on the original input. Our model can recognize 6 different document layouts: title, text, figure, caption, table and page. 

<img src="static/result/example.jpg" text-align:center alt="example image" width="500">  

Another docker is [MongoDB](https://www.mongodb.com/), which is a NoSQL database that can be used to store json-like documents. The MongoDB docker use to store our json files corresponding to the result image. The json file contains all information about the inference. For our example image above, the json file looks like this:
* labels:"{"text": 5, "figure": 1, "title": 3, "page": 1}"
* Image name:"example.jpg"
* Image path:"/home/appuser/detectron2_repo/code/static/result"  

The labels part including all the labels detected in our input image and the number of each label. Since we use MongoDB to store these data, if necessary, the inference results can do regular search.

## Model Training
Detectron2 provides a pre-trained model's weight we can simply use it though [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). Obviously, we cannot use this weight in our project directly. In order to make this weight suitable for our project, we need to prepare a training set and train this weight again. This process is also called transfer learning. In this project, we use [labelme](https://github.com/wkentaro/labelme) to label our data. We manually annotated 550 data and 6919 labels, 500 data for training and 50 data for test. Detectron2 provides a [colab tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) where you can easily train your own model with a free GPU. 
After 2000 iterations of training, the AP value of our model reach to **0.57**. 

Usually the weight provided by model zoo are more general, which means it probably not suitable for our project. Therefore, if we can find a model's target close to our project, the AP value will increase under the same training parameters. Fortunately, we found a similar [project](https://github.com/hpanwar08/detectron2). This project trained a powerful Mask-RCNN model for document layout analysis, and their AP value is close to 0.9. Of course, they paid a great price. They used the largest document layout dataset---[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) for training, this training set contains more than 100,000 images. For ordinary document layout analysis, this weight is powerful enough. But we hope to get a weight that is more suitable for our project, rather than a powerful weight. Therefore, we use our 500 training dataset to perform transfer learning again. 
Under the same training parameters, our AP value reached **0.6**. 

This AP value is acceptable, but there is no doubt that our AP value needs to be improved. Since this is an individual project, in order to speed up the progress, we will use this model weight in our project. The most direct and effective way to increase the AP value is to increase the number of training sets. We can label the data manually, just as we did before. But since we already got a weight, we can use this model weight to generate the training set. That is, we regared the inference result of this model as a ground truth. The inference of the model will undoubtedly have errors, because our model is not perfect (obviously, if our model is perfect, there is no need to continue training). But in this way, we can easily generate thousands of data for training. In addition to the AP value, we can also add noise to the training set to improve the robustness of our model. These will be part of the future work.


## Pipeline
You can choose to upload one or more document images, the input format should be **JPG**, or **PBM**. If you data is PDF, you can use [pdfimages](https://github.com/facebookresearch/detectron2) to convert. 

When the backend receives the input, the first step is to initialize our database to avoid errors due to missing result images. We will check whether the result images in the volume matches the json file in MongoDB. If the user has deleted some images in the volume (not recommended) or image lost, the corresponding json file in MongoDB will also be deleted. 

The next step is to check whether the input image already exists in our volume, if so, add this image path to a result list, and skip the inference part. This is to avoid repeated inferences and save resource.  
If we do not have the same image in our volume, then we enter the inference stage. At this stage, the input image is inferred by our model. After the inference, we will add this image to the result list.
The inference result is an annotated image and a json file containing the inference result. The image will be saved in the volume, and the json file will be saved in MongoDB. 
When all the input has been processed (found in the database, or completed inference), we will get a result list, which contains the path of all the result images. This list will be sent to the front end, HTML will read this list, and show the result. Since we use cv2 to read and save images. In order to avoid save errors, all resulting images will be converted to jpg format.

## Environment
The inference part don't requires CUDA. But if your device has a CUDA-based GPU, it will be enabled without modifying the code. Our test platform is Intel i7-8809g, without CUDA, inferring a single image takes around 7 seconds. 

If you want to train your own model, most models does not support training on the CPU.
