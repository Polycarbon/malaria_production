import logging
import os, sys
from threading import Thread
from imutils.video import FileVideoStream
import VideoInfo
from multiprocessing import Queue, Manager, Process

import cv2
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger('Preprocessor')


class Preprocessor(Thread):
    def __init__(self, manager, detector):
        Thread.__init__(self)
        self.file_name = manager.video_path
        self.fvs = FileVideoStream(self.file_name).start()
        self.dataFileName = self.file_name[:-4] + '.npy'

        self.flow_list = manager.flow_list
        self.onBufferReady = manager.onBufferReady
        self.detector = detector

    def run(self):
        log.info('start preprocess video...')
        frame_count = VideoInfo.FRAME_COUNT
        window_size = VideoInfo.WINDOW_SIZE
        step_size = VideoInfo.STEP_SIZE
        move_thres = 0.2
        buffer = []

        prev = self.fvs.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        buffer.append(prev_gray.astype("int16"))
        # TODO: Check why images bounding missing.
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

                curr = self.fvs.read()
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

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
                self.flow_list.append([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.put((frameId, buffer[-window_size:]))
                        self.detector.detectThread(self.onBufferReady)
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.put((frameId - 1, tmp[-window_size:]))
                        self.detector.detectThread(self.onBufferReady)
                    # clear buffer
                    tmp = []
                    buffer = []
                # Extract rotation angle
                # da = np.arctan2(H[1, 0], H[0, 0])
                # Move to next frame
                prev_gray = curr_gray
                buffer.append(prev_gray.astype("int16"))

            self.flow_list.append([0, 0])
            if len(buffer) >= step_size:
                self.onBufferReady.put((frameId, buffer[-window_size:]))
                self.detector.detectThread(self.onBufferReady)
                log.debug('last preprocess frame:fid-{} buffer-{}'.format(str(frameId), len(buffer[-window_size:])))
            # else:
            #     self.onBufferReady.put((frameId, None))
            #     self.detector.detectThread(self.onBufferReady)
            #     log.debug('last preprocess frame:fid-{} buffer-{}'.format(str(frameId),"None"))
            d = np.array(d)
            np.save(self.dataFileName, d)
        else:
            d = np.load(self.dataFileName)
            tmp = []
            for frameId in range(1, frame_count):
                # Detect feature points in previous frame
                (dx, dy) = d[frameId]
                curr = self.fvs.read()
                self.flow_list.append([dx, dy])
                if abs(dx) < move_thres and abs(dy) < move_thres:
                    buffer.append(prev_gray.astype("int16"))
                    if len(buffer) == window_size:
                        # send buffer to predict
                        self.onBufferReady.put((frameId, buffer[-window_size:]))
                        self.detector.detectThread(self.onBufferReady)
                        # step buffer
                        tmp = buffer[:step_size]
                        buffer = buffer[step_size:]
                else:
                    tmp.extend(buffer)
                    if len(tmp) >= window_size:
                        self.onBufferReady.put((frameId - 1, tmp[-window_size:]))
                        self.detector.detectThread(self.onBufferReady)
                    # clear buffer
                    tmp = []
                    buffer = []
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                prev_gray = curr_gray

            self.flow_list.append(d[0].tolist())
            if len(buffer) >= step_size:
                self.onBufferReady.put((frameId, buffer[-window_size:]))
                self.detector.detectThread(self.onBufferReady)
                log.debug('last preprocess frame:fid-{} buffer-{}'.format(str(frameId), len(buffer[-window_size:])))
        log.info('preprocess finished.')
