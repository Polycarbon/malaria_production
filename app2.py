import datetime, time
import os, logging
from multiprocessing import Process
from threading import Thread
from src.Management import Management
from src.DetectorThread import Detector
from src.PreprocessorThread import Preprocessor
from src.ObjectMapperThread import ObjectMapper

import flask
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from src.get_ip import get_ip

from model.load import init

PROPER_REGION = 0
RESNET = 1

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# initalize our flask app
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.config["UPLOAD_FOLDER"] = "videos"
ALLOWED_EXTENSIONS = {"mp4", "avi"}

# initalize ip address and path
# WARNING: should close firewall.
log.warning("should close firewall.")
SERVER_IP = (
    get_ip(interface="wifi")
    if get_ip(interface="wifi") is not None
    else "192.168.1.103"
)
PORT = 5000
SERVER_URL = "http://" + SERVER_IP + ":" + str(PORT)
STATIC_PATH = SERVER_URL + "/static/"

STATUS =["onWorking","Finished"]
#  "Finished"         
# "onWorking"         
# {"status": "Finished", "data": []}

# Change demo result 
DEMO_RESULT = {
                "status": "Finished",
                "data":[
                {
                    "count": 1,
                    "image": "http://192.168.1.3:5000/static/output/manual_5-movie-resize_00-37.png#1588318521941",
                    "time": "00-37"
                },
                {
                    "count": 1,
                    "image": "http://192.168.1.3:5000/static/output/manual_5-movie-resize_00-49.png#1588318521941",
                    "time": "00-49"
                },
                {
                    "count": 2,
                    "image": "http://192.168.1.3:5000/static/output/manual_5-movie-resize_01-13.png#1588318521941",
                    "time": "01-13"
                },
                {
                    "count": 1,
                    "image": "http://192.168.1.3:5000/static/output/manual_5-movie-resize_01-42.png#1588318521941","time": "01-42"
                }
            ]}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_respone(result_list):
    def f(res):
        res_dict = dict()
        time_now = int(time.time() * 1000) + 1

        res_dict["image"] = "/".join(
            [SERVER_URL, res.get("image") + "#" + str(time_now)]
        )
        res_dict["count"] = res.get("count")
        res_dict["time"] = res.get("time")
        return res_dict

    return {"status":STATUS[1],"data":list(map(f, result_list))}


@app.route("/getTest", methods=["GET"])
def getTest():
    time_now = int(time.time() * 1000) + 1
    data = {
        "data": "API GET Testing is working !!",
        "local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "UTC_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    
    return jsonify(DEMO_RESULT)

# @app.route("/upload", methods=["GET", "POST"])
def demo_upload():
    if request.method == "POST":
        # Upload files from Client
        if list(request.files.keys())[0] not in request.files:
            return "No file part"
        f = list(request.files.values())[0]
        if f.filename == "":
            return "No selected file"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            manager.init(video_path)
            # Process
            t = Thread(target = manager.test_set_finish, args = ())
            t.start()
            print(" file uploaded successfully ")
            isFinish = manager.get_finish()
            if isFinish: return jsonify(DEMO_RESULT)
            else: 
                progress = manager.get_progress()
                log.info("Progress: {}%".format(progress))
                return jsonify({"status":STATUS[0],"progress":progress})
                
    if request.method == "GET":
        isFinish = manager.get_finish()
        if isFinish:
            return jsonify(DEMO_RESULT)
        else: 
            progress = manager.get_progress()
            log.info("Progress: {}%".format(progress))
            return jsonify({"status":STATUS[0],"progress":progress})
                

@app.route("/upload", methods=["GET", "POST"])
def predict_upload():
    if request.method == "POST":
        # Upload files from Client
        if list(request.files.keys())[0] not in request.files:
            return "No file part"
        f = list(request.files.values())[0]
        if f.filename == "":
            return "No selected file"
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(video_path):
                os.remove(video_path)
            f.save(video_path)
            # return 'file uploaded successfully'

        # Predict Video
        # video_path = "videos/manual_5-movie-resize.mp4"
        manager.init(video_path)
        detector = Detector(
            manager=manager, mode=PROPER_REGION, model=model, graph=graph
        )
        ppc_worker = Preprocessor(manager, detector)
        map_worker = ObjectMapper(manager)

        ppc_worker.start()
        map_worker.start()

        # ppc_worker.join()
        # print("ppc_worker end")
        # map_worker.join()
        isFinish = manager.get_finish()
        if isFinish:
            manager.saveFile()
            res = get_respone(manager.get_result())
            log.info("data respone: {} - head(5):{}".format(len(res["data"]), res["data"][:5]))
            manager.cap_release()
            return jsonify(res)
        else:
            progress = manager.get_progress() 
            log.info("Progress: {}%".format(progress))
            return jsonify({"status":STATUS[0],"progress":progress})

    if request.method == "GET":
        isFinish = manager.get_finish()
        if isFinish:
            manager.saveFile()
            res = get_respone(manager.get_result())
            log.info("data respone: {} - head(5):{}".format(len(res["data"]), res["data"][:5]))
            manager.cap_release()
            return jsonify(res)
        else:
            progress = manager.get_progress() 
            log.info("Progress: {}%".format(progress))
            return jsonify({"status":STATUS[0],"progress":progress})

if __name__ == "__main__":
    manager = Management()
    model, graph = None,None #init()
    # decide what port to run the app in
    port = int(os.environ.get("PORT", PORT))
    # run the app locally on the givn port
    app.run(host=SERVER_IP, port=port)

    


# optional if we want to run in debugging mode
# app.run(debug=True)
