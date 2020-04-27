import datetime,time
import os,logging
from multiprocessing import Process
from src.Management import Management
from src.PreprocessorThread import Preprocessor
from src.ObjectMapperThread import ObjectMapper

import flask
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from src.get_ip import get_ip

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')

# initalize our flask app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = "videos"
ALLOWED_EXTENSIONS = {'mp4',"avi"}

# initalize ip address and path
# WARNING: should close firewall. 
SERVER_IP = get_ip(interface='wifi') if get_ip(interface='wifi') is not None else "192.168.1.103"
PORT = 5000
SERVER_URL = 'http://'+SERVER_IP+':'+str(PORT)
STATIC_PATH = SERVER_URL+'/static/'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_respone(result_list):
    def f(res):
        res_dict = dict()
        time_now = int(time.time() * 1000)+1
        
        res_dict["image"] = "/".join([SERVER_URL,res.get("image")+"#"+str(time_now)])
        res_dict["count"] = res.get("count")
        res_dict["time"] = res.get("time")
        return res_dict
    return {"data":list(map(f,result_list))}

@app.route('/getTest', methods=['GET'])
def getTest():
    time_now = int(time.time() * 1000)+1
    data = {"data": "API GET Testing is working !!", 
    "local_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "UTC_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    return jsonify(data)

@app.route('/upload', methods=['GET', 'POST'])
def predict_upload():
    if request.method == 'POST':
        # Upload files from Client
        if list(request.files.keys())[0] not in request.files: return 'No file part'
        f = list(request.files.values())[0]
        if f.filename == '': return 'No selected file'
        if f and allowed_file(f.filename):
            filename  = secure_filename(f.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename )
            if os.path.exists(video_path):
                os.remove(video_path)
            f.save(video_path)
            # return 'file uploaded successfully'
        
        # Predict Video
        # video_path = "videos/manual_5-movie-resize.mp4"
        Manager = Management(video_path)
        
        ppc_worker = Preprocessor(Manager)
        map_worker = ObjectMapper(Manager)

        ppc_worker.start()
        map_worker.start()

        # ppc_worker.join()
        # print("ppc_worker end")
        map_worker.join()

        Manager.saveFile()
        res = get_respone(Manager.get_result())
        log.info("data respone: {}- head(5):{}".format(len(res["data"]),res["data"][:5]))
        Manager.cap_release()
        return jsonify(res)

if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', PORT))
    # run the app locally on the givn port
    app.run(host=SERVER_IP, port=port)
    

# optional if we want to run in debugging mode
# app.run(debug=True)



