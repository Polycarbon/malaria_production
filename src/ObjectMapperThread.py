import logging
import os, time, sys
from collections import defaultdict
from threading import Thread
from multiprocessing import Queue, Manager, Process

import VideoInfo

from PyQt5.QtCore import QPointF, Qt, QPoint
from PyQt5.QtGui import QPolygonF
from ObjectHandler import ObjectTracker, CellRect
from centroidtracker import CentroidTracker

log = logging.getLogger('ObjMapper')


class ObjectMapper(Thread):
    def __init__(self, manager):
        Thread.__init__(self)
        self.frame_count = VideoInfo.FRAME_COUNT
        self.window_size = VideoInfo.WINDOW_SIZE
        self.curr_area = QPolygonF()
        self.objectmap = defaultdict(lambda: {'area': None, 'cells': []})
        self.currFrameId = 0
        self.Q = manager.Q
        self.set_finish = manager.set_finish
        self.tracker = ObjectTracker()
        self.flow_list = manager.flow_list
        self.focus_pt = QPointF(VideoInfo.FRAME_WIDTH / 2, VideoInfo.FRAME_HEIGHT / 2)
        self.curr_area_id = 0
        self.updateDetectLog = manager.updateDetectLog
        self.onUpdateProgress = manager.onUpdateProgress
        self.onObjectsMapUpdate = manager.updateFrameObjects

    def run(self):
        log.info('start ObjectMapper...')
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
                        if abs(x) > 50 or abs(y) > 50:
                            self.tracker.clear()
                        self.curr_area.translate(x, y)
                        cells = self.tracker.translated(x, y)
                        self.objectmap[i + 1] = {'area': None, 'cells': cells}
                        # self.objects.update(translated)

                    self.currFrameId = start_id

                if not self.curr_area.containsPoint(self.focus_pt, Qt.OddEvenFill):
                    self.last_area = self.curr_area
                    self.curr_area_id += 1
                    self.curr_area = QPolygonF(area_vec)
                else:
                    self.curr_area = QPolygonF(area_vec)

                self.curr_area = QPolygonF(area_vec)
                detected_cells = [CellRect(*cell) for cell in detected_cells]
                self.tracker.update(detected_cells)
                cells = self.tracker.getObjects()
                new_count, in_area_cells = self.tracker.countInArea(self.curr_area)
                # new and last conflict
                objects = {}
                self.objectmap[self.currFrameId] = {'area':  QPolygonF(self.curr_area), 'cells': cells}
                for i in range(self.currFrameId, end_id):
                    x, y = self.flow_list[i]
                    self.curr_area.translate(x, y)
                    cells = self.tracker.translated(x, y)
                    translated = dict([(k, cell.translated(x, y)) for k, cell in in_area_cells.items()])
                    in_area_cells.update(translated)
                    objects[i] = {'area': QPolygonF(self.curr_area), 'cells': translated}
                    self.objectmap[i+1] = {'area':  QPolygonF(self.curr_area), 'cells': cells}
                    # cells = self.tracker.translated(x, y)
                    # self.objects.update(translated)
                if new_count > 0:
                    self.updateDetectLog(self.currFrameId, self.curr_area, self.curr_area_id, in_area_cells, new_count,
                                         objects)
                self.currFrameId = end_id
                log.debug('frames {}-{} ,progress {}/{}'.format(start_id, end_id, end_id, self.frame_count))
                log.debug("  cell{} - scores{}".format(str(detected_cells), str(scores)))

                self.onUpdateProgress(self.currFrameId)

                if end_id == self.frame_count - 1:
                    time.sleep(1)
                    log.info('ObjectMapper finished')
                    self.set_finish(True)
                    self.onObjectsMapUpdate(self.objectmap)
                    return
