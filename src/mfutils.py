import cv2
import numpy as np
# from PyQt5.QtGui import QImage


""" def toQImage(image: np.ndarray):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
 """

def drawBoxes(image: np.ndarray,boxes, color=(0,255,0), thickness=2):
    for box in boxes:
        if isinstance(boxes,dict):
            cv2.rectangle(image,
                        (int(box.left()), int(box.top())),
                        (int(box.right()), int(box.bottom())), color,thickness)
        
        elif isinstance(boxes,list):
            cv2.rectangle(image,
                ((int(box[0])),(int(box[1]))),
                ((int(box[2])),(int(box[3]))),color,thickness)

def getHHMMSSFormat(ms):
    hour, remainder = divmod(ms, 3600000)
    min, remainder = divmod(remainder, 60000)
    sec, _ = divmod(remainder, 1000)
    return int(hour), int(min), int(sec)