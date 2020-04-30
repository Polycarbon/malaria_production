import cv2

from keras_retinanet.utils.image import preprocess_image, resize_image
from model.load import init
import numpy as np

# global model, graph
# # initialize these variables
# model = init()
# print("Loaded Model from disk")


# def detect(image):
#     image = preprocess_image(image)
#     image, scale = resize_image(image)
#     boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
#     boxes /= scale
#     return boxes[0], scores[0]


# img = cv2.imread('sample_input/2-7_sum0-50.png')
# cell_locs, scores = detect(img)

# print(cell_locs)
# print(scores)
