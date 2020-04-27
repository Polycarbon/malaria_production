from multiprocessing import Manager , Value
from ctypes import c_int , c_bool
import os,cv2,sys
import logging

sys.path.append("src/")
import VideoInfo
from mfutils import drawBoxes, getHHMMSSFormat

log = logging.getLogger('Management')

class Management:
    def __init__(self,video_path="",manager=None):
        self.manager = Manager()
        self.video_path = video_path
        self.Q = self.manager.Queue()
        self.flow_list = self.manager.list()
        self.onBufferReady = self.manager.Queue()
        
        self.result = self.manager.list()
        self.sum_cells = Value(c_int,0,lock=True)

        self.respone = list()

        if os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(video_path)
            VideoInfo.init_video(self.cap)
            self.frameCount = VideoInfo.FRAME_COUNT
            self.duration = VideoInfo.DURATION
    
    def updateDetectLog(self, detected_frame_id, cell_map, cell_count):
        # append log
        # widget = QCustomQWidget()
        self.sum_cells.value += cell_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, detected_frame_id)
        _, image = self.cap.read()
        _, min_, sec = getHHMMSSFormat(self.duration / self.frameCount * detected_frame_id * 1000)
        time_text = '{:02}-{:02}'.format(min_, sec)
        cell_map_list = list(map(lambda cell : cell.getCoords(),cell_map.values()))
        self.result.append({"image": image.copy(), "detect_time": time_text,"cells": cell_map_list})
        log.debug("result:{}".format(len(self.result)))
        # drawBoxes(image, cell_map, (0, 255, 0))
    
    def saveFile(self):
        log.info("start image saving...")
        # out = cv2.VideoWriter(self.vid_file_name, fourcc, frate, (fwidth, fheight))
        head, tail = os.path.split(self.video_path)
        out_dir = "static"
        file_prefix = tail.split('.')[0]
        self.respone = []
        for iter,l in enumerate(self.result):
            image = l['image']
            detect_time = l['detect_time']
            cells = l['cells']
            drawBoxes(image, cells, (0, 255, 0))
            file_name = "/".join([out_dir, file_prefix + "_" + detect_time + ".png"])
            cv2.imwrite(file_name, image)
            l.update({"image_path":file_name})
            
            self.respone.append({'image':file_name,"time":detect_time,"count":len(cells)})

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
    



# if __name__ == "__main__":
#     M = Management()
#     l = M.flow_list
#     l.append("55")