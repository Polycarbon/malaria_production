import cv2
import numpy as np

global FRAME_COUNT, FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_CENTER, WINDOW_SIZE, STEP_SIZE, DURATION
step = 0.5
window = 2  # sec

def init_video(cap):
    global FRAME_COUNT, FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_CENTER, WINDOW_SIZE, STEP_SIZE, DURATION
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = np.ceil(cap.get(cv2.CAP_PROP_FPS))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_CENTER = (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2))
    WINDOW_SIZE = int(FPS * window)
    STEP_SIZE = int(WINDOW_SIZE*step)
    DURATION = float(FRAME_COUNT) / float(FPS)