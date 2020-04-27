import datetime,time
import logging
from multiprocessing import Event, Manager, Process
from src.Management import Management
from src.PreprocessorThread import Preprocessor
from src.ObjectMapperThread import ObjectMapper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')

SERVER_IP = "http://localhost:5000"

def get_respone(result_list):
    def f(res):
        res_dict = dict()
        time_now = int(time.time() * 1000)+1
        
        res_dict["image"] = "/".join([SERVER_IP,res.get("image")+"#"+str(time_now)])
        res_dict["count"] = res.get("count")
        res_dict["time"] = res.get("time")
        return res_dict
    return list(map(f,result_list))

def APIupload():
    # ... upload complete ...
    video_path = "videos/manual_5-movie-resize.mp4"
    Manager = Management(video_path)
    
    ppc_worker = Preprocessor(Manager)
    map_worker = ObjectMapper(Manager)

    ppc_worker.start()
    map_worker.start()

    # ppc_worker.join()
    # print("ppc_worker end")

    map_worker.join()
    print("map_worker end")

    Manager.saveFile()
    res = get_respone(Manager.get_result())
    print(res)
    Manager.cap_release()
    return res
    

if __name__ == "__main__":
    APIupload()
    



