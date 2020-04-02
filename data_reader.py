import os
import constants
import cv2 as cv
from PIL import Image
import numpy as np
import tensorflow as tf

class ImageStream(object):
    def __init__(self, source_dir):
        self.feed_map = {x:os.path.join(source_dir, x) for x in os.listdir(source_dir)}
        self.feed = sorted([i for i in self.feed_map.keys()])
    
    def get_next(self):
        item = self.feed_map[self.feed.pop(0)]
        if item:
            item = Image.open(item)
            return np.array(item)
        
        return None
    
    def get_item(self, name = ""):
        item = self.feed_map[name]
        if item:
            # return Image.open(item)
            return cv.imread(item)

            # item = tf.io.read_file(item)
            # item = tf.image.decode_jpeg(item, channels=3)
            # item = tf.image.convert_image_dtype(item, tf.float32)
            # return item.numpy()
        
        return None
    
    def get_path_for_item(self, name):
        item = self.feed_map[name]
        if item and Image.open(item):
            return item

        return None

class VideoStream(object):
    def __init__(self, source):
        self.feed = cv2.VideoCapture(source)
        self.feed_tracker = 0
    
    def get_next(self):
        return self.feed.read()[1]

class DataStream(object):
    def __init__(self, source = constants.IMAGE_DIR):
        if os.path.isfile(source):
            self.feed = VideoStream(source)
        else:
            self.feed = ImageStream(source)

    
    def get_next(self):
        return self.feed.get_next()
    
    def get_item(self, name):
        return self.feed.get_item(name)
    
    def get_path_for_item(self, name):
        self.feed.get_path_for_item(name)
