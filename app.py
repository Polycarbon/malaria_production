# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

import base64
# system level operations (like loading files)
# for reading operating system data
import os
# for matrix math
# for importing our keras model
# for regular expressions, saves time dealing with string data
import re

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)
import cv2
from flask import Flask, render_template, request, jsonify
# scientific computing library for saving, reading, and resizing images
# from scipy.misc import imread, imresize

# tell our app where our saved model is
# sys.path.append(os.path.abspath("./model"))
# from model.load import *

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model


# initialize these variables
# model, graph = init()


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    print(imgData1)
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    imgdata = base64.b64decode(imgstr)
    # print(imgstr)
    with open('output.png', 'wb') as output:
        output.write(imgdata)


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/postRtsp', methods=['POST'])
def postRtsp():
    rtsp_address = request.json['endpoint']
    print(rtsp_address)
    vcap = cv2.VideoCapture(rtsp_address)
    print("success")
    i= 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        cv2.imshow('VIDEO', frame)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    vcap.release()
    cv2.destroyAllWindows()
    return jsonify(request.json), 201

@app.route('/getImage', methods=['GET'])
def getImage():
    data = {"image": "BASE64", "number": 5}
    return jsonify(data), 201


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    print("debug")
    # read the image into memory
    x = cv2.imread('output.png')

    print("debug2")
    x = preprocess_image(x)
    x, scale = resize_image(x)
    with graph.as_default():
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(x, axis=0))
    boxes /= scale
    data = []
    for i, (box, score) in enumerate(zip(boxes[0], scores[0])):
        # scores are sorted so we can break
        if score < 0.5:
            break
        data.append({'box': box.tolist(), 'score': float(score)})
    response = {'data': data}
    return response


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
# optional if we want to run in debugging mode
# app.run(debug=True)
