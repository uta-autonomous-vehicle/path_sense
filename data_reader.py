import os
import constants
import cv2

class ImageStream(object):
    def __init__(self, source_dir):
        self.feed = os.listdir(source_dir)
        self.feed = self.feed.sort()
        self.feed_tracker = 0
    
    def get_next(self):
        return self.feed.pop(0)

class VideoStream(object):
    def __init__(self, source):
        self.feed = cv2.VideoCapture(source)
        self.feed_tracker = 0
    
    def get_next(self):
        return self.feed.read()[1]

class DataStream(object):
    def __init__(self, source = constants.IMAGE_DIR):
        if os.path.isfile(source):
            self.feed = ImageStream(source)
        else:
            self.feed = VideoStream(source)

    
    def get_next(self):
        return self.feed.get_next()
