# our web app framework!

# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html
import logging
import base64
# system level operations (like loading files)
# for reading operating system data
import json
import os , time
from multiprocessing import Event
from queue import Queue
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
import numpy as np
import flask
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from lib.get_ip import get_ip

# scientific computing library for saving, reading, and resizing images
# from scipy.misc import imread, imresize

# tell our app where our saved model is
# sys.path.append(os.path.abspath("./model"))
from model.load import *

# PyQt
from MainWindow import MainWindow
from PyQt5.QtWidgets import QApplication

logging.basicConfig(format="%(threadName)s:%(message)s")
logger = logging.getLogger('data flow')
logger.setLevel(logging.DEBUG)

# initalize our flask app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = "./videos"
ALLOWED_EXTENSIONS = {'mp4',"avi"}

# initalize ip address and path
# WARNING: should close firewall. 
SERVER_IP = get_ip(interface='wifi') if get_ip(interface='wifi') is not None else "192.168.1.103"
PORT = 5000
STATIC_PATH = 'http://'+SERVER_IP+':'+str(PORT)+'/static/'

# global vars for easy reusability
global model
global number

# initialize these variables
model, graph = init()
number = 0

#
window = MainWindow()


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    print(imgData1[:100])
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    imgdata = base64.b64decode(imgstr)
    # print(imgdata[:100])
    with open('./static/output.png', 'wb') as output:
        output.write(imgdata)


def b64decode(image_path,save_path=None):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        decode_string = encoded_string.decode()
    URI_form = 'data:image/png;base64,'+decode_string
    
    if save_path not in [None,"",".","./"] :
        with open(save_path, "w") as txt:
            txt.write(URI_form)
    
    return URI_form , decode_string


def image_processing(buffer):
    # buffer = list(map(lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('int16'),buffer))
    v_max, v_min = 10, 1
    frameDiff = np.abs(np.diff(buffer, axis=0))
    frameDiffSum = np.sum(frameDiff, axis=0)
    av = (frameDiffSum / len(frameDiff))
    av[av > v_max] = v_max
    av[av < v_min] = v_min
    normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
    image = np.stack((normframe,) * 3, axis=-1)
    return image


def video_worker(RTSP_URL,viaThread=False,is_imshow=False):
    global number
    buffer = []
    background_image = None
    # RTSP_URL = rtsp://<IP>:<PORT>
    cap = cv2.VideoCapture(RTSP_URL)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    print("cap is on")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        if frame is not None:
            background_image = frame
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('int16')
            buffer.append(grayframe)
            if is_imshow: cv2.imshow('frame', frame)
        if is_imshow and cv2.waitKey(2) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    print("len(buffer):", len(buffer))
    if not viaThread: return buffer,background_image
    
    process_image = image_processing(buffer)
    pred = get_predict(process_image)
    boxes = pred["data"]
    number = pred["number"]
    image = get_draw_box_image(background_image,boxes)
    
    # cv2.imwrite("./static/process_image.png",process_image)
    cv2.imwrite("./static/output.png",image)
    print("predict:",pred)


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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")

@app.route('/postRtsp', methods=['POST'])
def postRtsp():
    """
    TODO: check processed images compare with postRtsp no Thread.

    FRONTEND: URL for image is <SERVER_IP>:<PORT>/static/output.png 
    len(buffer) ~= 214 , max ~=217 , min ~= 210
    """
    if request.method == 'POST':
        global number
        rtsp_url = request.json["endpoint"]
        worker = Thread(target=video_worker, args=(rtsp_url,True,False))
        worker.start()
        worker.join()
        print(request.json)

        time_now = int(time.time() * 1000)+1
        image_url = STATIC_PATH+'output.png#'+str(time_now)
        res = {'image': image_url,'number':number}
        print("POST response:",res)
        return jsonify(res)

# @app.route('/postRtsp', methods=['POST'])
def postRtsp_noThread():
    """
    TODO: check processed images compare with postRtsp via Thread.

    FRONTEND: URL for image is <SERVER_IP>:<PORT>/static/output.png 
    len(buffer) ~= 213-214 , max ~=217, min~=205 
    """
    if request.method == 'POST':
        global number
        rtsp_url = request.json["endpoint"]
        buffer,background_image = video_worker(rtsp_url)
        process_image = image_processing(buffer)

        # predict
        pred = get_predict(process_image)
        boxes = pred["data"]
        number= pred["number"]
        image = get_draw_box_image(background_image,boxes)

        # cv2.imwrite("./static/process_image.png",process_image)
        cv2.imwrite("./static/outputV2.png",image)
        print("predict:",pred)

        # URI_form,decode_base64 = b64decode("./static/outputV2.png")
        time_now = int(time.time() * 1000)+1
        image_url = STATIC_PATH+'outputV2.png#'+str(time_now)
        res = {'image': image_url,'number':number}
        print("POST response:",res)
        return jsonify(res)


@app.route('/getTest', methods=['GET'])
def getTest():
    time_now = int(time.time() * 1000)+1
    data = {"data": "API GET Testting is working !!", 
    "local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "UTC_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    return jsonify(data)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if list(request.files.keys())[0] not in request.files: return 'No file part'
        f = list(request.files.values())[0]
        if f.filename == '': return 'No selected file'
        if f and allowed_file(f.filename):
            filename  = secure_filename(f.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename )
            if not os.path.exists(video_path):
                f.save(video_path)
            # return 'file uploaded successfully'
            if False:
                worker = Thread(target=video_worker, args=(video_path,True,False))
                worker.start()
                worker.join()

                time_now = int(time.time() * 1000)+1
                image_url = STATIC_PATH+'output.png#'+str(time_now)
                res = {'image': image_url,'number':number}
                print("POST response:",res)
                return jsonify(res)
            
            else:
                terminate_event = Event()
                window.openFile(video_path,terminate_event)
                terminate_event.wait()
                log = window.saveFile()
                # window.closeEvent()
                print("POST response:",log)
                res = []
                for l in log:
                    detect_time = log['detect_time']
                    cells = log['cells']
                    file_name = log["image_path"]
                    res.append({"image":file_name,"count": cells,"time":detect_time})
                print("POST response:",res)
                return jsonify(res)

def psotTestUploadTime():
    pass



@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = cv2.imread('./static/output.png')
    # return jsonify({"data": [[1, 1, 1, 1], [2, 2, 2, 2]], "number": 2})
    return jsonify(get_predict(x))
    

if __name__ == "__main__":

    _ = QApplication([])
    # decide what port to run the app in
    port = int(os.environ.get('PORT', PORT))
    # run the app locally on the givn port
    app.run(host=SERVER_IP, port=port)
    

# optional if we want to run in debugging mode
# app.run(debug=True)
