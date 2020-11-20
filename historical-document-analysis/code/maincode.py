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
#model_weight = '/home/appuser/detectron2_repo/pub_model_final.pth'

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
    # 查找包含path name的所有不同结果
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

@app.route('/', methods=['POST', 'GET'])
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

@app.route('/search')
def search():
    '''
    这里要注意对于python来说$表示正则表达式匹配一个字符串的末尾，所以当Python遇到$就会去判断。
    所以要把这部分用引号括起来。
    同样的在MongoDB的语法中key的值是不需要引号的，但是在Python中不用引号会被当成变量。
    因此也要用引号。参见：https://stackoverflow.com/questions/28267831/passing-mongo-aggregation-to-python
    '''

    name = request.args.get('name')

    # 对于较为复杂的情况，可以通过aggregate实现
    # https://www.yangyanxing.com/article/aggregate_in_pymongo.html
    #match = {'image_name':{$regex:name,$options:"$i"}}
    #res_list = []
    # pipeline = [
    #     {'$match':{'image_name':{'$regex':name,'$options':'$i'}}},
    #     {'$group':{'_id':'$image_name','res_list':{'$push':'$path_name'}}}

    # res返回根据name找到的所有结果，结果是list，不区分大小写。
    # 这里设置全局变量，否则search filter取不到。
    global res, res_path
    res = mongo.db.result.distinct('image_name', {'image_name':{'$regex':name,'$options':"$i"}})
    res_no = len(res)

    if res_no == 0:
        return render_template('search_fail.html',keyword=name)

    res_path = []
    for i in res:
        search_result_path = os.path.join('static/result', i)
        res_path.append(search_result_path)

    return render_template('search_found.html',keyword=name,res_NO=res_no,res_list=res,image_list=Markup(res_path))

@app.route('/search_filter')
def search_filter():
    search_name = res
    search_path = res_path
    # 注意，这里不能用get因为传入的label都有相同的name，用get只能取到第一个值。
    labels = request.args.getlist('labels')

    # 对于用户传入的每一个label检查包含这个label的文件名，结果存到res_filter_all里面。
    res_filter_all = []
    for label in labels:
        user_input = 'labels.' + label
        res_tmp = mongo.db.result.distinct('image_name', {user_input:{'$exists':'true'}})
        res_filter_all.append(res_tmp)
    
    # 对res_filter_all里面的每一项取并集。这里的intersection只能对set有用，list不行。
    res_uni = set(res_filter_all[0])
    for i in range(1, len(res_filter_all)):
        res_uni = set.intersection(res_uni, set(res_filter_all[i]))
    
    res_uni_image = list(set.intersection(res_uni, set(search_name)))

    filter_path = []
    for i in res_uni_image:
        filter_tmp_path = os.path.join('static/result', i)
        filter_path.append(filter_tmp_path)
        
    return render_template('search_filter.html',res_list=search_name, search_labels=labels,res_labels=list(res_uni),inter=res_uni_image,image_list=Markup(filter_path))


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
