from flask import Flask, Markup, render_template, request, redirect, url_for, make_response,jsonify

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():

    return render_template('start.html',warning = "Illegal input, please choose again.")

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)