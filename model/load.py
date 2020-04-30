import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import numpy as np

global model, graph

def init():
    model = models.load_model('model/resnet50v2conf_67.h5', backbone_name='resnet50')
    print("Loaded Model from disk")
    graph = tf.get_default_graph()
    return model,graph
