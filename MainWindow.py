import VideoInfo
import logging
from Detector import CellDetector
import cv2
import imutils,os
import numpy as np
from Worker import PreprocessThread, ObjectMapper
from mfutils import drawBoxes, getHHMMSSFormat
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
logger = logging.getLogger('data flow')

class MainWindow():
    def __init__(self):
        self.input_name = None
        self.detector = CellDetector()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.mediaPlayer.setVideoOutput(self.videoWidget.videoSurface())

        self.mediaPlayer.setMuted(True)
    
    def startProcess(self):
        if self.input_name:
            self.sum_cells = 0
            self.log = []
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.input_name)))
            self.cap = cv2.VideoCapture(self.input_name)
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            VideoInfo.init(self.cap)
            map_worker = ObjectMapper()
            map_worker.onUpdateObject.connect(self.updateObject)
            # map_worker.onUpdateProgress.connect(dialog.updateProgress)
            ppc_worker = PreprocessThread(self.input_name)
            ppc_worker.onFrameChanged.connect(map_worker.updateOpticalFlow)
            ppc_worker.onBufferReady.connect(self.detector.detect)
            # ppc_worker.onUpdateProgress.connect(dialog.updateProgress)
            map_worker.onNewDetectedCells.connect(self.updateDetectLog)
            # map_worker.finished.connect(dialog.close)
            # dialog.closed.connect(ppc_worker.quit)
            # dialog.closed.connect(map_worker.quit)
            self.detector.onDetectSuccess.connect(map_worker.queueOutput)
            # map_worker.onfinished.connect(self.on_finished)
            # dialog.onReady2Read.connect(self.setOutput)
            # dialog.show()
            map_worker.start()
            ppc_worker.start()
    
    def on_finished(self,isCome=0):
        self.terminate_event.clear()
        print('thread finished')
        
            
    # TODO: USE THIS VIA API UPLOAD
    def openFile(self,file_name,terminate_event):
        # file_name = 
        self.terminate_event = terminate_event
        self.terminate_event.set()
        if os.path.exists(file_name):
            self.input_name = file_name
            self.startProcess()        

    # write iamge to static output
    def saveFile(self):
        
        # out = cv2.VideoWriter(self.vid_file_name, fourcc, frate, (fwidth, fheight))
        head, tail = os.path.split(self.input_name)
        out_dir = "static"
        file_prefix = tail.split('.')[0]

        for iter,log in enumerate(self.log):
            image = log['image']
            detect_time = log['detect_time']
            cells = log['cells']
            drawBoxes(image, cells, (0, 255, 0))
            file_name = os.path.join(out_dir, file_prefix + "_" + detect_time + ".png")
            cv2.imwrite(file_name, image)
            self.log[iter]["image_path"] = file_name
        # out.release()
        logger.info("save finish")
        return self.log
    
    def closeEvent(self, *args, **kwargs):
        QApplication.closeAllWindows()
        
    
    def updateObject(self, frame_objects):
        self.frame_objects = frame_objects
        duration = self.mediaPlayer.duration()
        # self.videoWidget.setOutput(frame_objects, duration / self.frameCount)
        # self.ui.playButton.setEnabled(True)
        # self.ui.saveButton.setEnabled(True)
    
    def updateDetectLog(self, detected_frame_id, cell_map, cell_count):
        # append log
        # widget = QCustomQWidget()
        self.sum_cells += cell_count
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, detected_frame_id)
        _, image = self.cap.read()
        _, min, sec = getHHMMSSFormat(self.mediaPlayer.duration() / self.frameCount * detected_frame_id)
        time_text = '{:02}-{:02}'.format(min, sec)
        self.log.append({"image": image.copy(), "detect_time": time_text, "cells": cell_map})
        drawBoxes(image, cell_map, (0, 255, 0))