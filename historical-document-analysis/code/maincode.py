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
app.secret_key = 'some_secret' # 没有设置
#mongo = PyMongo(app, uri="mongodb://localhost:27017/result")  # 本地测试时使用
mongo = PyMongo(app, uri="mongodb://mongodb:27017/result")  # 创建数据库实例，collection=result

# 从指定路径读取模型的yaml文件和权重文件，其中权重在启动build.sh时通过wget下载，yaml文件在GitHub仓库中已经包含。
model = "/home/appuser/detectron2_repo/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml"
model_weight = '/home/appuser/detectron2_repo/code/pub_model_final.pth'

# 设置上传文件尺寸，单个BSON文件限制最大为16M。超过16M用GridFS来存储和恢复，将大文件分割成多个小的chunk（256k/个）。
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 使得能够输出瑞典语, enable swedish（似乎不需要）。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置允许输入的图片文件格式。设定格式是为了防止跨站脚本攻击(XSS)。
# 当文件直接上传到客户端时，用户可能上传HTML文件，或者其他有害文件。
# https://dormousehole.readthedocs.io/en/latest/patterns/fileuploads.html
ALLOWED_EXTENSIONS = set(['JPG', 'JPGE', 'PBM'])

# rsplit 分割得到'.'后的文件拓展名。如果没有拓展名或者拓展名不在ALLOWED_EXTENSIONS则返回false。
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1]  # 只分割一次，将文件名分为两部分，[1]取拓展名。
    return '.' in filename and ext.upper() in ALLOWED_EXTENSIONS

# 检查MongoDB和实际的图片是否匹配，如果找不到对应的图片就把MongoDB内部的对应数据删除，并返回失效的list。
# 这是为了防止MongoDB内有图片和图片的labels字典，但是却没有在指定的路径上找到图片(用户误删，或其他原因)，这会使得前端无法显示结果。
def check_data_match():
    # 这里的result是DB里面的collection，如果初始化的时候不是这个名字那么每次调用MongoDB的时候都要改掉。
    # distinct返回path_name的所有不同结果(所有的path_name都应该是unique的，要是不出bug...)
    image_list = mongo.db.result.distinct('path_name')
    #print('path list:',image_list)
    invalid_image = []

    for image in image_list:
        exists = os.path.exists(image)
        if exists is not True:
            mongo.db.result.remove({"path_name":image}) # 没有在指定路径找到图片，删除MongoDB中path_name为image的文档
            invalid_image.append(image)
            print('this image not find:',image,'we will deleted')
    return invalid_image

# 根据输入名检查数据库中是否包含结果，如果没有返回false，之后再进行推断；如果有返回图片的路径，跳过推断。
# 因为有check_data_match()方法，因此如果MongoDB内部有图片路径，那么一定能找到这个图片。
def check_exist(input_name):
        # 这里检查图片名不能带拓展名，因为最后保存的结果是jpg格式的，而输入不一定是的。但是最后找图片的时候，必须带拓展。
        #os.path.splitext(input_name)[0] # 获取不带拓展的文件路径+文件名
        image_find = mongo.db.result.find_one({'image_name':input_name})
        if image_find is None:
            print('This image doesnot find:',input_name)
            return False
        else:
            image_path = image_find['image_path'] 
            return  image_path

@app.route('/', methods=['POST', 'GET'])
def upload():
    '''
    初始时返回start.html页面，当使用了POST方法之后返回result.html页面，支持POST多张图片。触发POST时，执行以下步骤：
    1. 定义保存图片的路径：当前路径+static/result。
    2. 调用check_data_match()检查路径中的图片结果和MongoDB的数据是否匹配。
    3. 检查输入的文件名是否合法。
    4. 循环检查输入是否在DB内。如果存在把路径添加到reslist，不存在先执行deteinfer，再把结果路径存进reslist，结果dict存入DB。
    5. 将存有所有结果路径的reslist传给result.html。
    '''
    if request.method == 'POST':
        # basepath 是当前maincode.py文件所在路径。file_path基于此路径建立一个储存推断结果image的文件夹。
        # 这个result文件夹是已经创建好的，如果没有则创建(不出bug不会)。
        basepath = os.path.dirname(__file__) 
        file_path = os.path.join(basepath, 'static/result')
        if not os.path.exists(file_path):
            os.makedirs(file_path, 755) # r读取=4，w写入=2，x执行=1。所有者r+w+x=7，同组用户r+x=5，公共用户r+x=5

        # 首先检查MongoDB中的记录和result中的图片文件是否对应
        invalid_image = check_data_match()
        if invalid_image:
            #flash('Those image not match!',invalid_image)
            print('check image match DB,those are not match:',invalid_image)
        else:
            print('check image match DB, all images matched. Go to next step')

        # 检查输入的文件扩展名是否非法。如果非法的拓展名，或者没有扩展名，返回false。
        # 输入可以是多个文件，getlist()内的名字'file'在start.html内post方法处定义。
        files = request.files.getlist('file')
        for file in files:
            if not (file and allowed_file(file.filename)):  # flask中使用file得到的不仅是文件名还包括其他信息，用file.filename获得文件名
                return render_template('start.html',warning = "Illegal input! Please use JPG or PBM file and keep the extension.")

        # 输入非空，并且合法，进入下一步。
        # 检查对应的名字是否在MongoDB，如果不在执行推断，如果在则直接返回路径。
        reslist = []
        for file in files:
            # 始终使用secure_filename，来检查文件名。这会把一些原本的../类型的字符修改。
            # “永远不要信任用户输入”，哪怕用户就是我自己(∩~∩)。
            image_name = secure_filename(file.filename)
            # 所有的结果都是保存为 jpg 格式的。
            check_name = os.path.splitext(image_name)[0]+'.jpg'
            #print('jpg input name:',check_name)
            check_result = check_exist(check_name)
            check_result_path = os.path.join('static/result', check_name) # 也是最后保存图片的位置
            #print('check result path:',check_result_path)
            
            # 如果在数据库中找到图片，打印文件名并返回路径。
            #if check_result is not False:
            if check_result:
                print('find result in our database, we do not need infer this image:',check_name,'\n')
                reslist.append(check_result_path)

            # 如果没有找到，首先保存上传的图片，然后推断。
            else:
                # 保存图片
                upload_path = os.path.join(file_path, image_name) # file_path = basepath+static/result
                file.save(upload_path)
                # print('input save,file path:',file_path,'file name:',image_name,'upload path:',upload_path)

                # 调用deteinfer中的infer方法。推断结果，并保存。
                # 该方法的返回值是结果图片的储存路径，和字典形式的推断结果标注文件。
                infer_path, json_res = deteinfer.infer(upload_path,model,model_weight)
                infer_name = os.path.split(infer_path)[1]
                infer_result_path = os.path.join('static/result', infer_name)
                print("done infer, result save in:",infer_result_path,'infer name:',infer_name,'\n')
                reslist.append(infer_result_path)  # 添加到结果列表

                # 这里保存为json之后会搜索不到，因此保存为字典。关闭ASCII避免瑞典语有乱码。
                #json_res = json.dumps(res_dic,ensure_ascii=False)
                # 将推断结果添加到 MongoDB，多个插入用insert_many()
                mongo.db.result.insert_one(json_res)

        # 这里的Markup很关键！！！不用的话这个list传递到JS后会根据ASCII转义表把数组的单引号转义为&#39，会报错。
        # 在jinja2中渲染模板的render_template会自动转义，因此要通过Markup来关闭转义，这样传给JS的才会是一个数组。
        # 但是出于安全原因，绝对不要对用户的直接输入用markup。
        return render_template('result.html', image_list=Markup(reslist))
    # 初始阶段，返回起始页面 start.html
    return render_template('start.html')

@app.route('/search')
def search():
    '''
    search方法，对于DB中已经保存了推断结果的情况，调用该方法来搜索指定结果。
    传入参数是用户想要查询的图片名（不包括路径，包括拓展名），返回值为所有的匹配结果。结果将可以实时预览，有多个结果时可以手动切页。
    所有的结果都被保存为jpg格式，因此用户可以输入jpg来获得数据库中的所有结果，并在前端预览。
    使用正则表达式来获得和输入关键词匹配的图片名，把找到的结果存入图片名数组res。
    如果res长度为0，说明无匹配结果，返回search_fail.html，传入参数为用户输入的搜索keyword。
    如果res长度非0，把所有res的路径添加到res_path。返回search_found.html，传入参数为keyword，结果数量res_no，图片名数组res和图片路径res_path。同时在前端显示结果。
    '''

    # name是search.HTML文件中定义的名字，它是一个由用户输入的搜索关键词。
    name = request.args.get('name')

    # 对于较为复杂的搜索，可以通过aggregate实现，这里不需要。
    # https://www.yangyanxing.com/article/aggregate_in_pymongo.html
    #match = {'image_name':{$regex:name,$options:"$i"}}
    #res_list = []
    # pipeline = [
    #     {'$match':{'image_name':{'$regex':name,'$options':'$i'}}},
    #     {'$group':{'_id':'$image_name','res_list':{'$push':'$path_name'}}}

    # res返回根据name找到的所有结果，结果是list，不区分大小写。res_path是所有res对应的路径。
    # 这里必须设置全局变量，否则之后的search filter方法取不到。
    global res, res_path

    # 这个正则表达式需要注意，对于pymongo来说$regex会被解释为变量。因此要把$部分用引号括起来，直接用MongoDB则不需要。
    # 参见：https://stackoverflow.com/questions/28267831/passing-mongo-aggregation-to-python
    # distinct语句找到所有image_name符合regex查询条件的结果，并且返回这些结果的image_name，因此res是一个仅包含图片名不包含路径的list。
    # 对括号内image_name字段的正则查询，image_name字段不包括图片路径，但是包括拓展名，因此用户查询'jpg'会返回数据库中所有的图片。option i表示不区分大小写。
    res = mongo.db.result.distinct('image_name', {'image_name':{'$regex':name,'$options':"$i"}})
    res_no = len(res)  # 找到的结果数量

    # 没有找到和关键字匹配的结果。返回search_fail.html页面，并传入用户的搜索关键词。前端显示：未搜索到XXX的结果。 
    if res_no == 0:
        return render_template('search_fail.html',keyword=name)

    # 因为所有的结果图片都存在同一路径，因此用join方法添加文件名，就能获得图片路径。
    res_path = []
    for i in res:
        search_result_path = os.path.join('static/result', i)
        res_path.append(search_result_path)
    # 传入前端需要的所有参数：用户查询的关键词；结果数量；结果列表；结果图片路径，注意Markup禁止转义。
    return render_template('search_found.html',keyword=name,res_NO=res_no,res_list=res,image_list=Markup(res_path))

@app.route('/search_filter')
def search_filter():
    '''
    search_filter方法对用户search的结果进行进一步过滤，过滤基于labels。本函数的启动代码在search_found.html里面。
    本函数在search页面中触发，因为只有找到了结果才能过滤。如果用户想要对数据库中的所有图片搜索，在search中输入jpg。
    本函数传入2个变量，search_name表示搜索结果图片名，search_path是搜索结果的路径。只有执行了search方法才会产生这两个数组。
    labels变量在本页由用户通过CheckBox的方式传入，用户选择希望结果中包含的labels，可多选。
    本函数返回search_filter.html页面。
    '''
    # 这里是 search 方法中定义的全局变量。
    search_name = res
    search_path = res_path
    # 注意，这里不能用get因为在前端代码中把所有的CheckBox的name都设为了labels，但是它们的value不同，用get只能取到第一个值。
    labels = request.args.getlist('labels')

    # 对于用户传入的每一个label，返回包含这个label的image_name，只要存在该label就返回，存在几个不重要。
    # res_tmp是一个数组，表示包含label的所有图片名。通过for循环把用户输入的每一个label都查询出来，然后写入res_filter_all（二维数组）里面。
    # 这里应该可以优化，但是没想好怎么弄。
    res_filter_all = []
    for label in labels:
        user_input = 'labels.' + label  # MongoDB中所有的label是保存在键labels下的一个字典，所以要用'labels.'+label
        res_tmp = mongo.db.result.distinct('image_name', {user_input:{'$exists':'true'}}) # 返回所有指定的label存在的文档的image_name
        res_filter_all.append(res_tmp)
    
    # 对res_filter_all里面的每一项取并集，所有用户选择的labels都要有。这里的intersection只能对set有用，list不行。
    res_uni = set(res_filter_all[0])
    for i in range(1, len(res_filter_all)):
        res_uni = set.intersection(res_uni, set(res_filter_all[i]))
    
    # 对search方法的搜索结果和这一步的过滤结果取并集。
    res_uni_image = list(set.intersection(res_uni, set(search_name)))
    
    # 获得所有结果的路径。
    filter_path = []
    for i in res_uni_image:
        filter_tmp_path = os.path.join('static/result', i)
        filter_path.append(filter_tmp_path)
    
    # 返回给前端的参数为：上一步search方法的所有结果；用户想要查询的labels；包含所有用户查询的label的图片名（由set转list）；search和labels的并集；结果图片路径。
    # 原则上如果filter为空，应该返回另一个页面类似于search_fail.html。算了，偷个懒。   
    return render_template('search_filter.html',res_list=search_name, search_labels=labels,res_labels=list(res_uni),inter=res_uni_image,image_list=Markup(filter_path))


if __name__ == '__main__':
    # host='0.0.0.0'对所有的IP地址开放，可以在Jenkins设置具体的允许访问IP。
    app.run(host='0.0.0.0', port=5000)
