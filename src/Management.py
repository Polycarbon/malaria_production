from multiprocessing import Manager , Value
from ctypes import c_int , c_bool
import os,cv2,sys,shutil,time
import logging

sys.path.append("src/")
import VideoInfo
from mfutils import drawBoxes, getHHMMSSFormat
import matplotlib.pyplot as plt
log = logging.getLogger('Management')
class Management:

    def __init__(self,manager=Manager):
        self.manager = manager()
        self.isfinish = Value(c_bool,False,lock=True)

    def init(self,video_path="",manager=None):
        self.video_path = video_path
        self.Q = self.manager.Queue()
        self.flow_list = self.manager.list()
        self.onBufferReady = self.manager.Queue()
        self.isfinish.value = False

        self.result = self.manager.list()
        self.sum_cells = Value(c_int,0,lock=True)

        self.respone = list()

        if os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(video_path)
            VideoInfo.init_video(self.cap)
            self.frameCount = VideoInfo.FRAME_COUNT
            self.duration = VideoInfo.DURATION
    
    def updateDetectLog(self, detected_frame_id, area_points, cell_map, cell_count):
        # append log
        self.sum_cells.value += cell_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, detected_frame_id+20)
        _, image = self.cap.read()
        _, min_, sec = getHHMMSSFormat(self.duration / self.frameCount * (detected_frame_id+20) * 1000)
        time_text = '{:02}-{:02}'.format(min_, sec)
        cell_map_list = list(map(lambda cell : cell.getCoords(),cell_map.values()))
        
        # draw counting area
        for i in range(area_points.size()-1):
            p1 = area_points.at(i)
            p2 = area_points.at(i+1)
            cv2.line(image,(int(p1.x()), int(p1.y())),(int(p2.x()), int(p2.y())),(255, 0, 0),2)
        
        # draw parasite cells
        drawBoxes(image, cell_map_list, (0,255, 0))

        self.result.append({"image": image.copy(), "detect_time": time_text,"cells": cell_map_list,"count":cell_count})
        log.debug("result:{}".format(len(self.result)))
    
    def saveFile(self,dir_path="static/output"):
        log.info("start image saving...")
        # out = cv2.VideoWriter(self.vid_file_name, fourcc, frate, (fwidth, fheight))
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
        os.mkdir(dir_path)
        
        head, tail = os.path.split(self.video_path)
        file_prefix = tail.split('.')[0]
        self.respone = []
        for iter,l in enumerate(self.result):
            image = l['image']
            detect_time = l['detect_time']
            cells = l['cells']
            count = l["count"]
            # drawBoxes(image, cells, (0, 255, 0))
            file_name = "/".join([dir_path, file_prefix + "_" + detect_time + ".png"])
            cv2.imwrite(file_name, image)
            l.update({"image_path":file_name})
            
            self.respone.append({'image':file_name,"time":detect_time,"count":count})

        log.info("save finish.")
    
    def get_result(self,keys=None):
        """ 
        keys list have 3 key: "image" , "time" , "count".
        """
        if not keys or keys is None: return self.respone.copy()
        else:
            def f(res):
                res_dict = dict()
                for k in keys: res_dict[k] = res.get(k)
                return res_dict
            return list(map(f,self.respone.copy()))
    
    def cap_release(self):
        self.cap.release()
    
    def test_set_finish(self,seconds=30):
        print("wait {} seconds...".format(seconds))
        time.sleep(seconds)
        self.isfinish.value = True 
        print("Finish !!")
    
    def set_finish(self,isfinish=False):
        self.isfinish.value = isfinish
        print("Set isFisnish:",self.isfinish.value)

    def get_finish(self):
        log.debug("isFisnish:",self.isfinish.value)
        return self.isfinish.value
    

# currFrameId / frame_count # inObjectMapper


# if __name__ == "__main__":
#     M = Management()
#     l = M.flow_list
#     l.append("55")