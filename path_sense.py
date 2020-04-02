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
            # NOTE: http://www.flatuicolorpicker.com/yellow-hsv-color-model
            processed.filter_color((60, 35, 35), (64, 100, 100))
            processed.display_image('small', 'original_image')
            # continue
            areas = processed.get_bounding_box()

            y,x = processed.image.shape
            processed.draw_line((x/2, 0), (x/2, y))
            
            if areas:
                bx,by,bw,bh = processed.get_closest_box(np.array(areas)[:,:4])
                x1, y1, x2, y2 = x/2, y, int(bx), int(by)
                processed.draw_line((x1,y1),(x2,y2))

                line1 = ((x1,y1),(x2,y2))
                line2 = ((x/2, 0), (x/2, y))
                print processed.angle_between_lines(line1, line2)

            processed.display_image('small', 'processed_image')
            CVTools(image).display_image('small', 'original_image_rgb')

            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break
            
            # time.sleep(0.3)
            print "**********"
        
        cv.destroyAllWindows()

detect_lines(0,0,0,0)