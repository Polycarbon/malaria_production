# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

import base64
# system level operations (like loading files)
# for reading operating system data
import json
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
from multiprocessing import Process
from threading import Thread
import queue
import numpy as np
import flask
from flask import Flask, render_template, request, jsonify, url_for

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
    print(imgData1[:100])
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    imgdata = base64.b64decode(imgstr)
    # print(imgdata[:100])
    with open('./static/output.png', 'wb') as output:
        output.write(imgdata)


def image_processing(buffer):
    # buffer = list(map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),buffer))
    v_max, v_min = 10, 1
    frameDiff = np.abs(np.diff(buffer, axis=0))
    frameDiffSum = np.sum(frameDiff, axis=0)
    av = (frameDiffSum / len(frameDiff))
    av[av > v_max] = v_max
    av[av < v_min] = v_min
    normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
    image = np.stack((normframe,) * 3, axis=-1)
    return image


def video_worker(RTSP_URL, is_imshow=False):
    buffer = []
    q = queue.Queue()
    # RTSP_URL = rtsp://<IP>:<PORT>
    cap = cv2.VideoCapture(RTSP_URL)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    print("cap is on")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        if is_imshow: cv2.imshow('frame', frame)
        buffer.append(frame)
        if is_imshow and cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("len(buffer):", len(buffer))
    return buffer


def get_predict(image):
    image = preprocess_image(image)
    image, scale = resize_image(image)
    with graph.as_default():
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    data = []
    for i, (box, score) in enumerate(zip(boxes[0], scores[0])):
        # scores are sorted so we can break
        if score < 0.5:
            break
        data.append({'box': box.tolist(), 'score': float(score)})
    response = {'data': data, 'number': len(data)}
    return response


def get_draw_box_image(image, bounding_box):
    for box in bounding_box:
        cv2.rectangle(image, (int(box["box"][0]), int(box["box"][1])),
                      (int(box["box"][2]), int(box["box"][3])),
                      (0, 255, 0), 2)
    return image


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/postRtsp', methods=['POST'])
def postRtsp():
    """
    global origin_image , process_image

    #json
    # rtsp_url = request.json["RTSP_URL"]

    # text
    rtsp_url = request.json

    buffer = get_buffer(rtsp_url)
    image = image_processing(buffer)
    process_image = image
    origin_image = buffer[3]

    """
    rtsp_url = request.json["endpoint"]
    # worker = Thread(target=video_worker, args=(rtsp_url, True))
    # worker.start()
    # worker.join()
    # print(request.json)
    res = {'image': 'http://192.168.1.103:5000/static/output.png'}
    return jsonify(res)


@app.route('/getImage', methods=['GET'])
def getImage():
    """
    pred = get_predict(process_image)
    boxes = pred["data"]
    image = get_draw_box_image(origin_image,boxes)

    filename = "output.png"
    cv2.imwrite("./static/"+filename, image)
    # encode = base64.b64encode(image)
    # decode = base64.decodebytes(encode)

    number = pred["number"]
    """
    data = {"data": "/static/output.png", "number": 2}
    return jsonify(data)
    # URL for image is <SERVER_IP>:<PORT>/static/output.png


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
    x = cv2.imread('./static/output.png')

    print("debug2")
    return jsonify({"data": [[1, 1, 1, 1], [2, 2, 2, 2]], "number": 2})


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
# optional if we want to run in debugging mode
# app.run(debug=True)
