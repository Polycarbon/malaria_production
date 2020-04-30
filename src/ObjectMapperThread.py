import logging
import os,time,sys
from threading import Thread
from multiprocessing import Queue, Manager, Process

import VideoInfo

from PyQt5.QtCore import QPointF, Qt, QPoint
from PyQt5.QtGui import QPolygonF
from ObjectHandler import ObjectTracker, CellRect
from centroidtracker import CentroidTracker

log = logging.getLogger('ObjMapper')

class ObjectMapper(Thread):
    def __init__(self,manager):
        Thread.__init__(self)
        self.frame_count = VideoInfo.FRAME_COUNT
        self.window_size = VideoInfo.WINDOW_SIZE
        self.curr_area = QPolygonF()
        self.currFrameId = 0
        self.Q = manager.Q
        self.updateDetectLog = manager.updateDetectLog
        self.tracker = ObjectTracker()
        self.flow_list = manager.flow_list
        self.focus_pt = QPointF(VideoInfo.FRAME_WIDTH / 2, VideoInfo.FRAME_HEIGHT / 2)
    
    # TODO fix bounding Only in Grid.
    #      updateDetectLog follow by grid per image not time_to_found per image.  
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
                        self.curr_area.translate(x, y)
                        cells = self.tracker.translated(x, y)
                        # self.objects.update(translated)

                    self.currFrameId = start_id

                if not self.curr_area.containsPoint(self.focus_pt, Qt.OddEvenFill):
                    self.last_area = self.curr_area
                    self.curr_area = QPolygonF(area_vec)
                else:
                    self.curr_area = QPolygonF(area_vec)

                self.curr_area = QPolygonF(area_vec)
                detected_cells = [CellRect(*cell) for cell in detected_cells]
                new_count = self.tracker.update(detected_cells)
                new_count = self.tracker.countInArea(self.curr_area.united(self.last_area))
                cells = self.tracker.getObjects()
                
                if new_count > 0:
                    self.updateDetectLog(self.currFrameId, cells, new_count)
                # new and last conflict
                for i in range(self.currFrameId, end_id):
                    x, y = self.flow_list[i]
                    # self.curr_area.translate(x, y)
                    cells = self.tracker.translated(x, y)
                    # self.objects.update(translated)
                self.currFrameId = end_id
                log.debug('frames {}-{} ,progress {}/{}'.format(start_id,end_id,end_id, self.frame_count))
                log.debug("  cell{} - scores{}".format(str(detected_cells), str(scores)))
                if end_id == self.frame_count - 1:
                    time.sleep(1)
                    log.info('ObjectMapper finished')
                    return 
            # log.debug('D')