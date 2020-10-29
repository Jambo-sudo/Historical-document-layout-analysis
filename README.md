# Document-layout-analysis

## Background
The purpose of this project is to build a document layout analysis model and interact with users through a webpage. Since this project mainly serves a historical literature research group, this model may not be suitable for other types of input images. But we can easily achieve this by using different models.

## Architecture
This project is mainly composed of two dockers. One is used to run inference part, and a flask framework to build the website and accept input data. This docker also mounts a volume, which contains all the inference results image. This volume can be bound to the local storage to achieve data persistence. Another docker is MongoDB, which is a NoSQL database that can be used to store json-like documents.

The model inference part uses [Mask-RCNN](https://arxiv.org/abs/1703.06870) and runs on the [detectron2](https://github.com/facebookresearch/detectron2) framework.The model runs in docker (currently on my computer). In addition, there is another MongoDB docker to store our json files, and a mounted volume to store the result images.The following figure shows a example of result image. We have marked out multiple layout divisions based on the original picture.Our model can recognize 6 different document layouts: title, text, figure, caption, table and page.

The json file contains more information about the inference. For our example, the json file looks like this:
* labels:"{"text": 5, "figure": 1, "title": 3, "page": 1}"</li>
* Image name:"1990-075.jpg"
* image path:"/home/appuser/detectron2_repo/code/static/result"
We use MongoDB to store these data, it means that if necessary, the inference results support regular search.


## Pipeline
You can choose to upload one or more document images, the format is JPG, or PBM. If you data is PDF, you can use [pdfimages](https://github.com/facebookresearch/detectron2) to convert. 
When the backend receives the input, it will check whether the result image in the volume matches the json file in MongoDB. If the user has deleted the image in the volume (not recommended) or lost, the corresponding file in MongoDB will also be been deleted.
The next step is to check whether the input image already exists in our volume, if so, add this image path to a result list, and skip the inference part. This is to avoid repeated inferences and save time.
If we do not have the same image in our volume, then we will enter the inference stage.At this stage, the input image inferred by the Mask-RCNN model. The inference result is an annotated image and a json file containing the inference result. The image will be saved in the volume, and the json file will be saved in MongoDB.We use cv2 to read and save images. In order to avoid save errors, all resulting images will be converted to jpg format.
