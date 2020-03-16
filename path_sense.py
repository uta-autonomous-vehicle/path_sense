import time
import os
import constants
import numpy as np
from PIL import Image
from data_reader import DataStream
import tensorflow as tf
import cv2 as cv

import pdb
from test import Follower
import math

from utils import CVTools

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

class LaneDetector(object):
    def __init__(self, data_set = ''):
        return
    
    def train(self, data_source):
        self.feed = DataStream(data_source)

    def test(self, data_source):
        self.feed = DataStream(data_source)

    def get_accuracy(self):
        return None 
    
    def get_loss(self):
        return None
    

def detect_lines(thres1, thres2, thresh3, thresh4):
    number_of_samples = 5
    base_dir = os.path.abspath(constants.IMAGE_DIR)
    dataset_dir = os.listdir(base_dir)
    for dataset_item in dataset_dir:
        item_path = os.path.join(base_dir, dataset_item)
        if not os.path.exists(os.path.join(item_path, 'left_camera')):
            # print "not a real path ", item_path
            continue
        
        data_stream = DataStream(os.path.join(item_path, 'left_camera'))
        raw_dataset = open( os.path.join(base_dir, dataset_item, 'left_camera.txt') )

        dataset = [[], []]
        line = raw_dataset.readline()

        while line:
            line = line.split(" ")
            
            image_name = line[0]
            steering = float(line[1])
            speed = float(line[2])
            
            # dataset[0].append(data_stream.get_item(image_name))
            # dataset[1].append(steering)
            line = raw_dataset.readline()

            image = data_stream.get_item(image_name)
            

            min_height = 720/3
            max_height = 720

            min_width = 0
            max_width = 1280/2

            image2 = np.zeros((720, 1280))

            processed = CVTools(image)
            processed.filter_color((30, 100, 100), (30, 255, 255))
            processed.get_bounding_box()

            y,x = processed.image.shape
            processed.draw_line((x/2, 0), (x/2, y))

            processed.display_image('large', '1')

            if cv.waitKey(1) == ord('q'):
                break
            if cv.waitKey(1) == ord('w'):
                time.sleep(1000.0)
            
            # time.sleep(0.3)

        print "***"

detect_lines(0,0,0,0)