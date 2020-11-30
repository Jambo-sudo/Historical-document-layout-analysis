# What in this folder
Static folder contains all static images, and static/result contains all inference result.  

Templates contain all HTML files, that is, our frontend.  

The deteinfer.py is the python file used to run detectron2 inference. If you need to modify the parameters of the inference model, you can modify it here.  

The maincode.py is our main file, which will be automatically launched when docker-compose is run. All major changes, such as DB, flask should be here.
