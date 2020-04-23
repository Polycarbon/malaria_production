import enum
import logging
import os
from queue import Queue

import cv2
from PyQt5.QtCore import QThread, QRect, QRectF, QPointF, Qt, QPoint
from PyQt5 import QtCore
from PyQt5.QtGui import QPolygonF
from PyQt5.QtWidgets import QApplication
import numpy as np
from scipy.spatial.distance import cdist

import VideoInfo
from ObjectHandler import ObjectTracker, CellRect
from centroidtracker import CentroidTracker
from keras_retinanet import models
from scipy.ndimage import binary_closing
from scipy.spatial import distance
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from imutils.video import FileVideoStream
from keras_retinanet.utils.image import preprocess_image, resize_image

logger = logging.getLogger('data flow')


window_time = 2


class PreprocessThread(QThread):
    onImageReady = QtCore.pyqtSignal(list, list, list, np.ndarray)
    # onBufferReady = QtCore.pyqtSignal(list, np.ndarray, list)
    onBufferReady = QtCore.pyqtSignal(int, list)
    onFinish = QtCore.pyqtSignal()
    onFrameChanged = QtCore.pyqtSignal(list)

    def __init__(self, file_name):
        QThread.__init__(self)
        self.fvs = FileVideoStream(file_name).start()
        self.dataFileName = file_name[:-4] + '.npy'
        self.binarySignal = None

    def __del__(self):
        self.wait()

    def run(self):
        QApplication.processEvents()
        logger.info('start preprocess video')
        frame_count = VideoInfo.FRAME_COUNT
        window_size = VideoInfo.WINDOW_SIZE
        step_size = VideoInfo.STEP_SIZE
        move_thres = 0.5
        buffer = []
        prev = self.fvs.read()
        # prev = cv2.resize(prev, targetSize)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        # prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        buffer.append(prev_gray.astype("int16"))
        # frame_count = 500
        if not os.path.exists(self.dataFileName):
            d = [(0, 0)]
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                                   maxCorners=200,
                                                   qualityLevel=0.01,
                                                   minDistance=30,
                                                   blockSize=100)
                # Read next frame
                curr = self.fvs.read()
                # Convert to grayscale
                # curr = cv2.resize(curr, targetSize)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
                # Calculate optical flow (i.e. track feature points)
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                # Sanity check
                assert prev_pts.shape == curr_pts.shape
                # Filter only valid points
                idx = np.where(status == 1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                # Find transformation matrix
                H, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
                # Extract traslation
                dx = H[0, 2]
                dy = H[1, 2]
                d.append([dx, dy])
                self.onFrameChanged.emit([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.emit(frameId, buffer[-window_size:])
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.emit(frameId - 1, tmp[-window_size:])
                    # clear buffer
                    tmp = []
                    buffer = []
                # Extract rotation angle
                # da = np.arctan2(H[1, 0], H[0, 0])
                # Move to next frame
                prev_gray = curr_gray
                buffer.append(prev_gray.astype("int16"))

            self.onFrameChanged.emit([0, 0])
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
            else:
                self.onBufferReady.emit(frameId, None)
            d = np.array(d)
            np.save(self.dataFileName, d)
        else:
            d = np.load(self.dataFileName)
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                (dx, dy) = d[frameId]
                curr = self.fvs.read()
                self.onFrameChanged.emit([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.emit(frameId, buffer[-window_size:])
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.emit(frameId - 1, tmp[-window_size:])
                    # clear buffer
                    tmp = []
                    buffer = []
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_gray = curr_gray

            self.onFrameChanged.emit(d[0].tolist())
            if len(buffer) >= step_size:
                self.onBufferReady.emit(frameId, buffer[-window_size:])
        logger.debug('preprocess finished')


class ObjectMapper(QThread):
    onfinished = QtCore.pyqtSignal(int)

    onUpdateObject = QtCore.pyqtSignal(defaultdict)
    onNewDetectedCells = QtCore.pyqtSignal(int, OrderedDict, int)

    def __init__(self):
        QThread.__init__(self)
        self.stopped_id = None
        self.frame_count = VideoInfo.FRAME_COUNT
        self.window_size = VideoInfo.WINDOW_SIZE
        self.objectmap = defaultdict(lambda: {'area': None, 'cells': []})
        self.curr_area = QPolygonF()
        self.last_cells = None
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.nextObjectID = 0
        self.lastFrameId = 0
        self.currFrameId = 0
        self.Q = Queue()
        self.tracker = ObjectTracker()
        self.flow_list = []
        self.objectId = 0
        self.focus_pt = QPointF(VideoInfo.FRAME_WIDTH / 2, VideoInfo.FRAME_HEIGHT / 2)

    def __del__(self):
        self.wait()

    def updateOpticalFlow(self, d):
        self.flow_list.append(d)

    def queueOutput(self, *args):
        print("sadasdsad")
        self.Q.put(args)
        # logger.debug('{}-{} : queue success'.format(args[0] - 50, args[0]))

    def run(self):
        logger.info('start ObjectMapper')
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.empty():
                (end_id, area_vec, detected_cells, scores) = self.Q.get()
                area_vec = list(map(lambda p: QPointF(*p), area_vec))
                start_id = int(end_id + 1 - self.window_size) if end_id + 1 > self.window_size else 0
                if self.currFrameId < start_id:
                    # get last cells
                    for i in range(self.currFrameId, start_id):
                        x, y = self.flow_list[i]
                        self.curr_area.translate(x, y)
                        cells = self.tracker.translated(x, y)
                        self.objectmap[i + 1] = {'area': None, 'cells': cells}
                        # self.objects.update(translated)

                    self.currFrameId = start_id

                if not self.curr_area.containsPoint(self.focus_pt, Qt.OddEvenFill):
                    self.last_area = self.curr_area
                    self.curr_area = QPolygonF(area_vec)
                else:
                    self.curr_area = QPolygonF(area_vec)

                self.curr_area = QPolygonF(area_vec)
                detected_cells = [CellRect(*cell) for cell in detected_cells]
                # last_cells = self.objectmap[self.currFrameId]["cells"]
                new_count = self.tracker.update(detected_cells)
                new_count = self.tracker.countInArea(self.curr_area.united(self.last_area))
                cells = self.tracker.getObjects()
                if new_count > 0:
                    self.onNewDetectedCells.emit(self.currFrameId, cells, new_count)
                # new and last conflict
                self.objectmap[self.currFrameId] = {'area': self.curr_area.united(self.last_area), 'cells': cells}
                for i in range(self.currFrameId, end_id):
                    x, y = self.flow_list[i]
                    # self.curr_area.translate(x, y)
                    cells = self.tracker.translated(x, y)
                    self.objectmap[i + 1] = {'area': self.curr_area.united(self.last_area), 'cells': cells}
                    # self.objects.update(translated)
                self.currFrameId = end_id
                self.onUpdateObject.emit(self.objectmap)
                logger.info('frames {}-{} ,progress {}/{}'.format(start_id,end_id,end_id, self.frame_count))
                logger.info("  cell{} - scores{}".format(str(detected_cells), str(scores)))
                if end_id == self.frame_count - 1:
                    self.sleep(1)
                    logger.info('ObjectMapper finished')
                    # self.onfinish.emit("sdfdsf")
                    return 
            # logger.debug('D')
