# coding:utf-8
from flask import Flask, Markup, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import sys
from PIL import Image 
import os, json, cv2, io
import numpy as np
import torch
from flask_pymongo import PyMongo
import deteinfer

app = Flask(__name__)
app.secret_key = 'some_secret'
#mongo = PyMongo(app, uri="mongodb://localhost:27017/result")  # 开启数据库实例
mongo = PyMongo(app, uri="mongodb://mongodb:27017/result")  


model = "/home/appuser/detectron2_repo/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml"
model_weight = '/home/appuser/detectron2_repo/code/pub_model_final.pth'

# 使得能够输出瑞典语
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['JPG', 'JPGE', 'PBM'])
 
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1]
    return '.' in filename and ext.upper() in ALLOWED_EXTENSIONS

# 设置静态文件缓存过期时间
#app.send_file_max_age_default = timedelta(seconds=1)

# 检查MongoDB和图片是否匹配，如果找不到对应的图片就把MongoDB内部的对应数据删除。
def check_data_match():
    image_list = mongo.db.result.distinct('path_name')
    #print('path list:',image_list)
    invalid_image = []

    for image in image_list:
        exists = os.path.exists(image)
        if exists is not True:
            mongo.db.result.remove({"path_name":image})
            invalid_image.append(image)
            print('this image not find:',image,'we will deleted')

    return invalid_image


def check_exist(input_name):
        # 这里的result是DB里面的collection，如果不是这个名字记得改掉
        # 这里检查名字不能带拓展名，因为最后保存的结果是jpg格式的，而输入不一定是的。但是最后找图片的时候，必须带拓展。
        #os.path.splitext(input_name)[0]
        image_find = mongo.db.result.find_one({'image_name':input_name})
        if image_find is None:
            print('This image doesnot find:',input_name)
            return False
        else:
            image_path = image_find['image_path'] 
            return  image_path

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        # 根据当前文件所在路径，创建一个储存image的文件夹
        basepath = os.path.dirname(__file__) 
        file_path = os.path.join(basepath, 'static/result')
        if not os.path.exists(file_path):
            os.makedirs(file_path, 755)

        reslist = []
        # 首先检查json和图片是否对应
        invalid_image = check_data_match()
        if invalid_image:
            #print('all data match!')
            #flash('Those image not match!',invalid_image)
            print('check image match,those are not match:',invalid_image)
        else:
            print('check image match, all images matched json')

        files = request.files.getlist('file')
        for file in files:
        # 如果非法的拓展名，或者为空，或者没有.那么返回error。
            if not (file and allowed_file(file.filename)):
                return render_template('start.html',warning = "Illegal input! Please use JPG or PBM file.")
        
        # 检查对应的名字是否存在，如果不在继续推断，如果在则直接返回。
        for file in files:
            image_name = secure_filename(file.filename)

            check_name = os.path.splitext(image_name)[0]+'.jpg'
            #print('jpg input name:',check_name)
            check_result = check_exist(check_name)
            check_result_path = os.path.join('static/result', check_name)
            #print('check result path:',check_result_path)
            if check_result is not False:
                print('find result in our database!:',check_name,'\n')
                reslist.append(check_result_path)

            else:
                # 保存图片
                upload_path = os.path.join(file_path, image_name)
                file.save(upload_path)
                # print('input save,file path:',file_path,'file name:',image_name,'upload path:',upload_path)

                # 推断结果，并保存
                infer_path, json_res = deteinfer.infer(upload_path,model,model_weight)
                infer_name = os.path.split(infer_path)[1]
                infer_result_path = os.path.join('static/result', infer_name)
                print("done infer,infer path:",infer_result_path,'infer name:',infer_name,'\n')
                reslist.append(infer_result_path)

                # 把json写入数据库
                mongo.db.result.insert_one(json_res)

                # print('done this file, reslist is:',reslist)

        return render_template('result.html', image_list=Markup(reslist))
 
    return render_template('start.html')
 
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)