# coding:utf-8
'''
这部分代码是用来执行模型推断的，调用接口为：deteinfer.infer(upload_path,model,model_weight)
输入参数分别为，input图片地址，使用模型的yaml文件和模型权重。
返回内容是，标注好的结果图片(与输入图片同名)，包含图片路径和classes信息的字典。
通常返回的结果图片被保存在result文件夹内，图片标注的字典则被存到MongoDB中。
'''
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import os, json, cv2, random, io
from collections import Counter
from PIL import Image

MetadataCatalog.get("dla_train").thing_classes = ['caption', 'figure', 'page', 'table', 'title', 'text']

# 原始的输出图片没有class labels，通过这个方法添加labels。
# 输入为预测的结果，和保存路径。输出为一个标记好了标签的字典。
def convert_dic(pred_class, path):
  classes = ['caption', 'figure', 'page', 'table', 'title', 'text']

  # convert the predicte class result to numpy
  result_np = pred_class.data.cpu().numpy()

  # return a dic, the key is class (but is a number) and the value is the number of this class in the result.
  count = Counter(result_np)
  # 针对数据写入问题，修改了以下代码。先把counter的结果变为字典.然后把结果res_dic也保留为字典。
  # 之前在MongoDB中找不到labels是因为把new dic变为了json，所以最后的res中labels对应的结果是一个str而不是字典，这样就找不到了。
  # 其他的class都能找到，因为它们都是字典。直接找label也能找到，因为这也是一个字典。
  count = dict(count)  
  new_dic = count.copy()

  # use the classes name, replace the number
  for i in count.keys():
    ind = int(i)
    new_key = classes[ind]
    new_dic[new_key] = new_dic.pop(i)

  # new_dic = json.dumps(new_dic,ensure_ascii=False)
  # res_dic = {'labels':new_dic}

  #new_dic = json.dumps(new_dic,ensure_ascii=False)
  res_dic = {}
  res_dic['labels'] = new_dic
    # 后加入的
  #res_dic = json.dumps(res_dic)

  filename = os.path.split(path)[1]
  filepath = os.path.split(path)[0]

  res_dic['image_name'] = filename
  res_dic['image_path'] = filepath
  res_dic['path_name'] = path

  # json化之后不好再修改，因此最好到存入DB之前再做，关闭ASCII否则瑞典语有乱码(似乎不需要转json)
  #json_res = json.dumps(res_dic,ensure_ascii=False)
  # print('json file result:',json_res)
  return res_dic
  #return json_res

# 模型推断部分。
# 输入是一个图片的地址，模型的yaml文件和权重。
# 输出为一张推断好的图片，包含了图片标注文件的字典（json之后存入MongoDB）。
def infer(input_path,model,model_weight):
  #im = input_path
  #这里的im需要是一张图片，因此如果是图片路径就需要先通过imread变成图片，如果是url就需要通过load方法。
  im = cv2.imread(input_path)

  # 获取模型的配置文件。
  cfg = get_cfg()
  cfg.merge_from_file(model)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this mode
  cfg.MODEL.WEIGHTS = model_weight
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

  # 判断cuda是否可用。推断不需要cuda，但是训练模型需要。
  if torch.cuda.is_available():
    print('now we use cuda')
    cfg.MODEL.DEVICE='cuda'
  else:
    print('running on cpu')
    cfg.MODEL.DEVICE='cpu'

  # 开始预测，如果只要坐标或者分类，可以用outputs["instances"]获得。
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  #outputs["instances"].pred_classes
  #outputs["instances"].pred_boxes
  pred_class = outputs["instances"].pred_classes

  # 把预测的结果可视化在原图上。
  v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  outs = out.get_image()[:, :, ::-1]

# 将input的原始文件删除，因为推断结果会使用同一个名字。
# 文件被限定为jpg文件，非jpg似乎有奇怪的bug。另外training set也是jpg格式。
  if os.path.exists(input_path):
      os.remove(input_path)
      print('remove input image')

# 如果输入的是jpg文件，则直接保存，否则转为jpg格式。
  if os.path.splitext(input_path)[-1] == ".jpg":
      cv2.imwrite(input_path,outs)
      print('input is a jpg file:',input_path)
      return input_path,convert_dic(pred_class,input_path)
  else:
    image_name = os.path.splitext(os.path.split(input_path)[1])[0]
    print('image name:',image_name)
    jpg_name = os.path.join(os.path.split(input_path)[0],image_name+'.jpg')
    cv2.imwrite(jpg_name,outs)
    print('convert input to jpg:',jpg_name)
    return jpg_name,convert_dic(pred_class,jpg_name)