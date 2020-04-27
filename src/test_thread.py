import os
import time
from collections import defaultdict, OrderedDict
from queue import Queue
from threading import Thread

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, QObject, QCoreApplication, Qt, QPointF
import numpy as np
from PyQt5.QtGui import QPolygonF
from imutils.video import FileVideoStream
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing

import VideoInfo
from LineHandler import calculateBoundingPoints, extend_horizontals, extend_verticals, extractLines
from ObjectHandler import CellRect, ObjectTracker

global reader, mapper, detector


class Reader(QThread):
    onBufferReady = QtCore.pyqtSignal(int, list)
    onFrameChanged = QtCore.pyqtSignal(list)

    def __init__(self):
        QThread.__init__(self)
        self.fvs = None
        self.dataFileName = None

    def __del__(self):
        self.wait()

    def setVideoPath(self,path):
        self.fvs = FileVideoStream(path).start()
        self.dataFileName = path[:-4] + '.npy'

    def run(self):
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


class Detector(QObject):
    onDetectSuccess = QtCore.pyqtSignal(int, list, list, list)

    def __init__(self, parent=None):
        super(Detector, self).__init__(parent)
        self.model = None
        self.thread = QThread()
        self.thread.start()
        self.moveToThread(self.thread)

    def detect(self, cur_frame_id, buffer):
        v_max = 10
        v_min = 1
        verticals, horizontals = extractLines(buffer[0], threshold=0.66)
        shape = buffer[0].shape
        center = (int(shape[1] / 2), int(shape[0] / 2))
        x_bound = [0, shape[1]]
        y_bound = [0, shape[0]]
        vs = extend_verticals(verticals, x_bound, y_bound)
        hs = extend_horizontals(horizontals, x_bound, y_bound)
        area_vec = calculateBoundingPoints(center, vs, hs)
        # area_vec = self.find_count_area(buffer[0])

        frameDiff = np.abs(np.diff(buffer, axis=0))
        frameDiffSum = np.sum(frameDiff, axis=0)
        av = (frameDiffSum / len(frameDiff))
        av[av > v_max] = v_max
        av[av < v_min] = v_min
        normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')

        image = normframe
        thresh = threshold_yen(image)
        binary = image >= thresh
        closed = binary_closing(binary)
        label_img = label(closed)
        cell_locs = regionprops(label_img)
        # tlbr to ltrb
        cells = []
        for cell in cell_locs:
            t, l, b, r = cell.bbox
            if 100 < cell.area:
                cells.append([l, t, r - l, b - t])
        if len(cells) > 5:
            # print("num{} move{}".format(len(cells),move_distances))
            self.onDetectSuccess.emit(cur_frame_id, area_vec, [], [])
            return
        cells.extend(cells)
        cells, weights = cv2.groupRectangles(cells, 1, 1.0)
        sc = [1.0] * len(cells)
        self.onDetectSuccess.emit(cur_frame_id, area_vec, list(cells), sc)


class Mapper(QThread):
    onObjectUpdate = QtCore.pyqtSignal(int)
    onNewDetectedCells = QtCore.pyqtSignal(int, OrderedDict, int)

    def __init__(self):
        QThread.__init__(self)
        self.stopped_id = None
        self.frame_count = VideoInfo.FRAME_COUNT
        self.window_size = VideoInfo.WINDOW_SIZE
        self.objectmap = defaultdict(lambda: {'area': None, 'cells': []})
        self.curr_area = QPolygonF()
        self.objects = OrderedDict()
        self.currFrameId = 0
        self.Q = Queue()
        self.tracker = ObjectTracker()
        self.flow_list = []
        self.objectId = 0
        self.focus_pt = QPointF(VideoInfo.FRAME_WIDTH / 2, VideoInfo.FRAME_HEIGHT / 2)

    def __del__(self):
        self.wait()

    def updateOpticalFlow(self, flow):
        self.flow_list.append(flow)

    def queueOutput(self, *args):
        self.Q.put(args)

    def run(self):
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
                self.currFrameId = end_id
                if end_id == self.frame_count - 1:
                    self.sleep(1)
                    return


def readLog(frameId, cells, new_count):
    print("id{} count{} :".format(frameId,len(cells)))


def startEventLoop():
    global reader, mapper, detector
    app = QCoreApplication([])
    path = "videos/manual_5-movie-resize.mp4"
    cap = cv2.VideoCapture(path)
    VideoInfo.init_video(cap)
    reader = Reader()
    mapper = Mapper()
    detector = Detector()
    reader.onBufferReady.connect(detector.detect)
    reader.onFrameChanged.connect(mapper.updateOpticalFlow)
    detector.onDetectSuccess.connect(mapper.queueOutput)
    mapper.onNewDetectedCells.connect(readLog)
    app.exec()


def start_task(path):
    global reader, mapper, detector
    cap = cv2.VideoCapture(path)
    VideoInfo.init_video(cap)
    reader.setVideoPath(path)
    reader.start()
    mapper.start()


if __name__ == "__main__":
    qcoreThread = Thread(target=startEventLoop)
    video_path = "videos/manual_5-movie-resize.mp4"
    qcoreThread.start()
    print("do something")
    time.sleep(5)
    start_task(video_path)
    qcoreThread.join()
