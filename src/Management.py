from multiprocessing import Manager, Value
from threading import Thread
from ctypes import c_int, c_bool
import os, cv2, sys, shutil, time
import imageio
from pygifsicle import optimize
import logging

sys.path.append("src/")
import VideoInfo
from mfutils import drawBoxes, getHHMMSSFormat
# import matplotlib.pyplot as plt

log = logging.getLogger('Management')


class Management:

    def __init__(self, manager=Manager, output_path="static/output"):
        self.manager = manager()
        self.progress = 0
        self.isfinish = Value(c_bool, False, lock=True)
        self.output_path = output_path

    def init(self, video_path="", manager=None):
        self.video_path = video_path
        self.Q = self.manager.Queue()
        self.flow_list = self.manager.list()
        self.onBufferReady = self.manager.Queue()
        self.isfinish.value = False
        self.result = self.manager.list()
        self.sum_cells = Value(c_int, 0, lock=True)

        self.respone = list()

        self.progress = 0

        if os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(video_path)
            VideoInfo.init_video(self.cap)
            self.frameCount = VideoInfo.FRAME_COUNT
            self.duration = VideoInfo.DURATION

            head, tail = os.path.split(self.video_path)
            self.file_prefix = tail.split('.')[0]

        # clear folder output
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=True)
        os.mkdir(self.output_path)

    #TODO optimize parameter
    def updateDetectLog(self, detected_frame_id, area_points, curr_area_id, cell_map, cell_count, objects):
        # append log
        self.sum_cells.value += cell_count
        _, min_, sec = getHHMMSSFormat(self.duration / self.frameCount * (detected_frame_id + 20) * 1000)
        time_text = '{:02}-{:02}'.format(min_, sec)
        cell_map_list = list(map(lambda cell: cell.getCoords(), cell_map.values()))
        buffer = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, detected_frame_id)
        for obj in objects.values():
            _, image = self.cap.read()
            area_points = obj["area"]
            # draw counting area
            for i in range(area_points.size() - 1):
                p1 = area_points.at(i)
                p2 = area_points.at(i + 1)
                cv2.line(image, (int(p1.x()), int(p1.y())), (int(p2.x()), int(p2.y())), (255, 0, 0), 2)

            # draw parasite cells
            cells = obj["cells"]
            drawBoxes(image, cells, (0, 255, 0))
            RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            buffer.append(RGB_image)

        """ 
        Concept
        A)  check areapoint is equal and flow_list ?
            check cells in area if new cells -> append it
        B)  append GridID
        """

        # save GIF if saveFile is `Slow`
        gif_name = "/".join([self.output_path, self.file_prefix + "_" + time_text + ".gif"])
        self.save_gif(gif_name,buffer)

        #  append in result list 
        self.result.append(
            {"buffer": buffer.copy(), "detect_time": time_text, "cells": cell_map_list,
             "count": cell_count})
        log.debug("result:{}".format(len(self.result)))

    def save_gif(self, gif_name, buffer):
        def f(gif_name, buffer):
            imageio.mimwrite(gif_name, buffer)
            optimize(gif_name)
        thread = Thread(target=f, args=(gif_name, buffer))
        thread.start()

    def saveFile(self, dir_path="static/output"):
        log.info("start image saving...")

        # if os.path.exists(dir_path):
        #     shutil.rmtree(dir_path, ignore_errors=True)
        # os.mkdir(dir_path)
        # os.mkdir(gif_path)

        head, tail = os.path.split(self.video_path)
        file_prefix = tail.split('.')[0]
        self.respone = []
        for iter, l in enumerate(self.result):
            buffer = l["buffer"]
            detect_time = l['detect_time']
            cells = l['cells']
            count = l["count"]

            # save image
            # file_name = "/".join([dir_path, file_prefix + "_" + detect_time + ".png"])
            # cv2.imwrite(file_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # l.update({"image_path":file_name})

            # save GIF
            gif_name = "/".join([dir_path, file_prefix + "_" + detect_time + ".gif"])
            # self.save_gif(gif_name, buffer)
            l.update({"gif_path": gif_name})

            self.respone.append({'gif': gif_name, "time": detect_time.replace("-", ":"), "count": count})

        log.info("save finish.")

    def get_result(self, keys=None):
        """ 
        keys list have 4 key: "gif", "time" , "count".
        """
        if not keys or keys is None:
            return self.respone.copy()
        else:
            def f(res):
                res_dict = dict()
                for k in keys: res_dict[k] = res.get(k)
                return res_dict
            return list(map(f, self.respone.copy()))

    def cap_release(self):
        self.cap.release()

    def test_set_finish(self, seconds=30):
        self.progress = 60
        print("wait {} seconds...".format(seconds))
        time.sleep(seconds)
        self.progress = 100
        self.isfinish.value = True
        print("Finish !!")

    def set_finish(self, isfinish=False):
        self.isfinish.value = isfinish
        print("Set isFisnish:", self.isfinish.value)

    def get_finish(self):
        log.debug("isFisnish:{}".format(self.isfinish.value))
        return self.isfinish.value

    def onUpdateProgress(self, currFrameId):
        self.progress = int(currFrameId / (self.frameCount - 1) * 100)
        log.debug("Progress({}/{}):{}%".format(currFrameId, self.frameCount - 1, self.progress))

    def get_progress(self):
        return self.progress

# if __name__ == "__main__":
#     M = Management()
#     l = M.flow_list
#     l.append("55")
