from multiprocessing import Manager, Value
from threading import Thread
from ctypes import c_int, c_bool
import os, cv2, sys, shutil, time
import numpy as np
import ffmpeg_streaming
import imageio
from pygifsicle import optimize
import logging

from src import VideoInfo

sys.path.append("src/")

from src.mfutils import drawBoxes, getHHMMSSFormat
# import matplotlib.pyplot as plt

log = logging.getLogger('Management')


class Management:
    PROCESSING = "Processing"
    SAVE_IMAGE = "Save Image"
    SAVE_VIDEO = "Save Video"
    UPLOAD_VIDEO = "Upload Video"
    FINISHED = "Finished"
    STATUS = [PROCESSING, FINISHED]

    def __init__(self, manager=Manager, output_path="static/output"):
        self.manager = manager()
        self.progress = 0
        self.isfinish = Value(c_bool, False, lock=True)
        self.output_path = output_path
        self.curr_status = None
        self.gif_path = "/".join([output_path, "GIFs"])
        self.media_path = "/".join([output_path, "media"])

    def init(self, video_path="", manager=None):
        self.video_path = video_path
        self.curr_status = self.PROCESSING
        self.video_out_path = None
        self.stream_path = None
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
        os.mkdir(self.media_path)
        os.mkdir(self.gif_path)

    # TODO optimize parameter
    def updateDetectLog(self, detected_frame_id, area_points, curr_area_id, cell_map, cell_count, objects):
        self.sum_cells.value += cell_count
        time_ms = self.duration / self.frameCount * detected_frame_id * 1000
        _, min_, sec = getHHMMSSFormat(time_ms)
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
            if int(np.sum(RGB_image)) > 0: 
                buffer.append(RGB_image)

        # save GIF if saveFile is `Slow`
        # head, tail = os.path.split(self.video_path)
        # file_prefix = tail.split('.')[0]
        # gif_name = "/".join([self.gif_path, file_prefix + "_" + detect_time + ".gif"])
        # self.save_gif(gif_name,buffer)

        #  append in result list
        if len(buffer) > 0:
            self.result.append(
                {"buffer": buffer.copy(), "detect_time": time_text, "detect_time_ms": int(time_ms), "cells": cell_map_list,
                "count": cell_count, "grid_id": curr_area_id})
            log.debug("result:{}".format(len(self.result)))

    def save_gif(self, gif_name, buffer):
        def f(gif_name, buffer):
            try:
                imageio.mimwrite(gif_name, buffer)
                optimize(gif_name)
            except RuntimeError:
                print("gif:{},(len:{}),{}".format(gif_name,len(buffer),np.sum(buffer, axis = 0)))
                
        thread = Thread(target=f, args=(gif_name, buffer))
        thread.start()

    def saveFile(self, dir_path="static/output"):
        log.info("start image saving...")
        gif_path = "/".join([dir_path, "GIFs"])

        head, tail = os.path.split(self.video_path)
        file_prefix = tail.split('.')[0]
        self.respone = []
        grid_set = set([v['grid_id'] for v in self.result])
        for i, grid_id in enumerate(grid_set):
            self.progress = round((i+1) / len(grid_set) * 100)
            grid_logs = list(filter(lambda d_log: d_log['grid_id'] == grid_id, self.result))
            detect_time = grid_logs[0]['detect_time']
            time_ms = grid_logs[0]['detect_time_ms']
            buffer = grid_logs[-1]["buffer"]
            count = 0
            for l in grid_logs:
                count += l['count']
            gif_name = "/".join([gif_path, file_prefix + "_" + detect_time + ".gif"])
            self.save_gif(gif_name, buffer)
            self.respone.append({'gif': gif_name, "time": detect_time.replace("-", ":"),"time_ms":time_ms, "count": count})

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
        pass
        # self.isfinish.value = isfinish
        # print("Set isFisnish:", self.isfinish.value)

    def updateFrameObjects(self, frame_objects):
        # assert len(frame_objects) == VideoInfo.FRAME_COUNT ,\
        #     str(len(frame_objects))+'=='+str(VideoInfo.FRAME_COUNT)
        head, tail = os.path.split(self.video_path)
        file_prefix = tail.split('.')[0]
        media_path = "/".join([self.output_path, "media"])
        self.video_out_path = "/".join([media_path, file_prefix + ".mp4"])
        self.stream_path = "/".join([media_path, file_prefix + ".m3u8"])
        self.curr_status = self.SAVE_IMAGE
        self.saveFile()
        if os.path.exists(self.stream_path):
            self.isfinish.value = True
            self.curr_status = self.FINISHED
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fwidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        flenght = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frate = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(self.video_out_path, fourcc, frate, (fwidth, fheight))
        self.curr_status = self.SAVE_VIDEO
        self.progress = 0
        for i in range(0, flenght):
            ret, frame = self.cap.read()
            cells = frame_objects[i]["cells"]
            area = frame_objects[i]["area"]
            drawBoxes(frame, cells, (0, 255, 0))

            if area:
                for j in range(area.size() - 1):
                    p1 = area.at(j).toPoint()
                    p2 = area.at(j + 1).toPoint()
                    cv2.line(frame, (p1.x(), p1.y()), (p2.x(), p2.y()), (255, 0, 0), 1)

            out.write(frame)
            self.progress = int((i+1) /flenght * 100)
        out.release()

        self.curr_status = self.UPLOAD_VIDEO
        self.stream_path = "/".join([media_path, file_prefix + ".m3u8"])
        video = ffmpeg_streaming.input(self.video_out_path)
        hls = video.hls(ffmpeg_streaming.Formats.h264())
        hls.auto_generate_representations()

        def monitor(ffmpeg, duration, time_):
            self.progress = round(time_ / duration * 100)

        hls.output(self.stream_path, monitor=monitor)
        self.isfinish.value = True
        self.curr_status = self.FINISHED
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
