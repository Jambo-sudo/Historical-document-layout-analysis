# coding:utf-8
'''
本方法是用来执行模型推断的，调用方式为：deteinfer.infer(upload_path,model,model_weight)
输入参数分别为：待推断的图片地址（upload_path），要使用的模型的yaml文件（model）和该模型的权重（model_weight）。
返回结果有两个：标注好的结果图片(与输入图片同名，后缀统一为jpg)；包含图片路径和labels信息的字典（由convertDict方法实现）。
通常返回的结果图片被保存在static/result文件夹内，返回的结果将包含路径。图片labels的字典则应该被保存到MongoDB中，这需要在maincode中实现。
注意：当推断完成后，原始的输入图片将被删除。
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

# MetadataCatalog.get("dla_train").thing_classes = ['caption', 'figure', 'page', 'table', 'title', 'text']

'''
给原始的结果添加class名字，并创建要传给MongoDB的字典。
输入为predictor预测的所有class的结果，和包括路径的结果图片。
输出为字典，key包括：labels：所有识别出来的labels和对应的数量（字典）；image_name：文件名；image_path：文件路径；path_name：文件路径+文件名。
'''
def convertDict(pred_class, path):
  classes = ['caption', 'figure', 'page', 'table', 'title', 'text']

  # convert the predicte class result to numpy
  # .data提取结果中的tensor，.cpu()用CPU来处理张量，numpy()转为numpy格式。
  result_np = pred_class.data.cpu().numpy()

  # return a dic, the key is class (but is a number) and the value is the number of this class in the result.
  # 统计结果中各个class的出现次数保存为字典，但是这里的class是数字的形式，形如：1:3；2:2等。
  count = Counter(result_np)
  count = dict(count)  # 把 Counter 类型的结果转为普通的字典格式。
  new_dic = count.copy()

  # use the classes name, replace the number。把count中的键由数字改为class的名字。
  for i in count.keys():
    ind = int(i)
    new_key = classes[ind]
    new_dic[new_key] = new_dic.pop(i)

  # 针对之前的数据写入和读取问题，做了如下修改。
  # 先把counter的结果变为字典（37行），然后把结果res_dic也保留为字典，res_dic的labels键对应所有的找到的labels，即new_dic。
  # 之前在MongoDB中找不到labels是因为把new dic变为了json，所以最后的res中labels键对应的结果是一个str而不是字典，这样就找不到了。
  # 其他的class都能找到，因为它们都是字典。直接找label也能找到，因为这也是一个字典。
  # new_dic = json.dumps(new_dic,ensure_ascii=False)
  # res_dic = {'labels':new_dic}
  
  # res_dic是要保存到MongoDB中的文件，它是一个字典。
  # res_dic建立一个新的键labels，这个键的值是一个字典new_dic。之后找特定的label的时候要先找label，再找label的名字。
  res_dic = {}
  res_dic['labels'] = new_dic 

  filename = os.path.split(path)[1]
  filepath = os.path.split(path)[0]
  
  # 给 res_dic 添加其他的键：文件名，路径，文件名+路径（用来直接给前端显示结果）。
  res_dic['image_name'] = filename
  res_dic['image_path'] = filepath
  res_dic['path_name'] = path

  # json化之后不好再修改，因此最好到存入DB之前再做，关闭ASCII否则瑞典语有乱码(从实验结果来看不转json，直接给字典也行)
  #json_res = json.dumps(res_dic,ensure_ascii=False)
  #print('json file result:',json_res)
  #return json_res
  return res_dic

'''
模型的推断部分。
输入包括：包含路径的图片地址，模型的yaml文件和模型的权重。
输出为一张推断好的图片，包含了图片标注信息的json字典（之后可以存入MongoDB）。
'''
def infer(input_path,model,model_weight):
  #im = input_path
  #这里的im需要是一张图片，因此如果是图片路径就需要先通过imread变成图片，如果是url就需要通过load方法（暂时不考虑）。
  im = cv2.imread(input_path)

  # 配置模型的主要参数。
  cfg = get_cfg()
  cfg.merge_from_file(model)    # 根据输入的model文件构造模型
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model
  cfg.MODEL.WEIGHTS = model_weight    # 根据输入的权重来配置模型权重 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6   # 分类数量为6个

  # 判断cuda是否可用。本模型的推断不需要cuda，但是训练模型需要。
  if torch.cuda.is_available():
    print('now we use cuda')
    cfg.MODEL.DEVICE='cuda'
  else:
    print('running on cpu')
    cfg.MODEL.DEVICE='cpu'

  # 开始预测，如果只要坐标或者分类，可以用outputs["instances"]获得。
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  #outputs["instances"].pred_classes  # 输出模型的分类
  #outputs["instances"].pred_boxes    # 输出分类的坐标
  pred_class = outputs["instances"].pred_classes

  # 把预测的结果输出给 outs，如果是测试要直接打印可以用imshow方法。
  v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  outs = out.get_image()[:, :, ::-1]
  # cv2_imshow(out.get_image()[:, :, ::-1]) # 显示图片

  # 将input的原始文件删除，因为推断结果会使用同一个名字。
  # 结果图片被限定为jpg文件，非jpg似乎有奇怪的bug。另外模型的training set也主要是jpg格式。
  if os.path.exists(input_path):
      os.remove(input_path)
      print('remove input image')

  # 如果输入的是jpg文件，则直接保存，否则转为jpg格式。
  if os.path.splitext(input_path)[-1] == ".jpg":  # splitext 分割文件名+拓展名
      cv2.imwrite(input_path,outs)
      print('input is a jpg file:',input_path)
      return input_path,convertDict(pred_class,input_path)
  else:
    image_name = os.path.splitext(os.path.split(input_path)[1])[0] # split 分割路径+文件名（包含拓展名）
    print('image name:',image_name)  # image_name是不带拓展和路径的文件名
    jpg_name = os.path.join(os.path.split(input_path)[0], image_name+'.jpg')
    cv2.imwrite(jpg_name, outs)
    print('convert input to jpg:',jpg_name)
    return jpg_name,convertDict(pred_class,jpg_name)
