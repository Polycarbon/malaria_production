import logging,sys

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread
import numpy as np
from scipy.spatial import distance
from skimage import feature, exposure
from skimage.feature import blob_dog, blob_doh
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, dilation, square, erosion

import VideoInfo
from LineHandler import extractLines, calculateBoundingPoints, extend_verticals, extend_horizontals

sys.path.append(".")
from model.load import *
# from keras_retinanet import models
# from keras_retinanet.utils.image import resize_image, preprocess_image

logger = logging.getLogger('data flow')

PROPER_REGION = 0
RESNET = 1
BLOB = 2


class CellDetector(QObject):
    onInitModelSuccess = QtCore.pyqtSignal()
    onDetectSuccess = QtCore.pyqtSignal(int, list, list, list)

    def __init__(self, parent=None):
        super(CellDetector, self).__init__(parent)
        self.model = None
        self.thread = QThread()
        self.thread.start()
        self.moveToThread(self.thread)
        self.mode = PROPER_REGION
        self.initModel()

    def initModel(self, path='model/resnet50v2conf_67.h5', backbone='resnet50'):
        if self.mode == RESNET:
            logger.info('Initialize Model')
            if self.model is None:
                # self.model = models.load_model(path, backbone_name=backbone)
                self.model,_ =  init()
                self.onInitModelSuccess.emit()
                logger.info('Init Model Success')
        else:
            logger.info('no need to initialize Model')
            self.onInitModelSuccess.emit()

    def setMode(self, mode):
        if mode == 1:
            self.mode = RESNET
        else:
            self.mode = PROPER_REGION

    # def setObjectMap(self, objectmap):
    #     self.objectmap = objectmap

    def detect(self, cur_frame_id, buffer):
        v_max = 10
        v_min = 1
        # flow_list = np.array(self.flow_list[cur_frame_id - 50:cur_frame_id]).transpose()
        # move_distances = np.sum(np.sqrt(flow_list[0] ** 2 + flow_list[1] ** 2))
        # if move_distances > 5:
        #     self.onDetectSuccess.emit(cur_frame_id,[], [], [])
        #     return
        verticals, horizontals = extractLines(buffer[0], threshold=0.66)
        shape = buffer[0].shape
        center = (int(shape[1] / 2), int(shape[0] / 2))
        x_bound = [0, shape[1]]
        y_bound = [0, shape[0]]
        vs = extend_verticals(verticals, x_bound, y_bound)
        hs = extend_horizontals(horizontals, x_bound, y_bound)
        area_vec = calculateBoundingPoints(center, vs, hs)

        frameDiff = np.abs(np.diff(buffer, axis=0))
        frameDiffSum = np.sum(frameDiff, axis=0)
        av = (frameDiffSum / len(frameDiff))
        av[av > v_max] = v_max
        av[av < v_min] = v_min
        normframe = (((av - v_min) / (v_max - v_min)) * 255).astype('uint8')
        if self.mode == RESNET:
            # preprocess
            image = np.stack((normframe,) * 3, axis=-1)
            image = preprocess_image(image)
            image, scale = resize_image(image)
            boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale
            cells = []
            sc = []
            for cell, score in zip(boxes[0], scores[0]):
                if score > 0.5:
                    l, t, r, b = cell
                    cells.append([int(l), int(t), int(r - l), int(b - t)])
                    # cells.append(cell)
                    sc.append(score)
            # min cluster size = 2, min distance = 0.5:
            cells.extend(cells)
            cells, weights = cv2.groupRectangles(cells, 1, 1.0)
            if len(cells) == 0:
                self.onDetectSuccess.emit(cur_frame_id, area_vec, [], [])
                return
            self.onDetectSuccess.emit(cur_frame_id, area_vec, cells.tolist(), sc)
        
        if self.mode == PROPER_REGION:
            # logger.info('start detect')   
            image = normframe
            thresh = threshold_yen(image)
            binary = image >= thresh
            closed = binary_closing(binary)
            # plt.title("move distance:{}".format(move_distances))
            # plt.imshow(closed)
            # plt.show()
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
            logger.info('detect finished {} , {}'.format(str(cur_frame_id),len(buffer)))   
