# Historical document layout analysis

## Background
The purpose of this project is to build a document layout analysis model and interact with users through a webpage. Since this project mainly serves a historical literature research group, this model may not be suitable for other types of input images. But we can easily deal with other tasks by using different models.

## Architecture
This project is mainly composed of two dockers. One docker including the model inference, a flask framework to build the website and accept input data. This docker also mounts a volume, which contains all the inference results image. This volume can be bound to the local storage to achieve data persistence. The model inference part uses [Mask-RCNN](https://arxiv.org/abs/1703.06870) and runs on the [detectron2](https://github.com/facebookresearch/detectron2) framework. 

The following figure shows a example of result image. The model marked out multiple layout divisions based on the original input. Our model can recognize 6 different document layouts: title, text, figure, caption, table and page. 

<img src="static/result/example.jpg" text-align:center alt="example image" width="500">  

Another docker is [MongoDB](https://www.mongodb.com/), which is a NoSQL database that can be used to store json-like documents. The MongoDB docker use to store our json files corresponding to the result image. The json file contains all information about the inference. For our example image above, the json file looks like this:
* labels:"{"text": 5, "figure": 1, "title": 3, "page": 1}"
* Image name:"1990-075.jpg"
* Image path:"/home/appuser/detectron2_repo/code/static/result"  

The labels part including all the labels detected in our input image and the number of each label. Since we use MongoDB to store these data, if necessary, the inference results can do regular search.

## Pipeline
You can choose to upload one or more document images, the input format should be JPG, or PBM. If you data is PDF, you can use [pdfimages](https://github.com/facebookresearch/detectron2) to convert. 

When the backend receives the input, the first step is initial the dataset. We will check whether the result image in the volume matches the json file in MongoDB. If the user has deleted some images in the volume (not recommended) or image lost, the corresponding json file in MongoDB will also be deleted.

The next step is to check whether the input image already exists in our volume, if so, add this image path to a result list, and skip the inference part. This is to avoid repeated inferences and save resource.  
If we do not have the same image in our volume, then we enter the inference stage. At this stage, the input image inferred by our model. The inference result is an annotated image and a json file containing the inference result. The image will be saved in the volume, and the json file will be saved in MongoDB. We use cv2 to read and save images. In order to avoid save errors, all resulting images will be converted to jpg format.

## Environment
The inference part don't requires CUDA. But if your device has a CUDA-based GPU, it will be enabled without modifying the code. Our test platform is Intel i7-8809g, without CUDA, inferring a single image takes around 7 seconds. 

If you want to train your own model, it usually does not support training on the CPU.
