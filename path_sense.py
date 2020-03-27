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
from utils.logger import logger

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

    # creating windows before hand
    # cv.namedWindow("processing_boxing", cv.WINDOW_NORMAL)
    # cv.namedWindow("processed_image", cv.WINDOW_NORMAL)
    # cv.namedWindow("original_image_rgb", cv.WINDOW_NORMAL)
    # cv.namedWindow("original_image", cv.WINDOW_NORMAL)



    for dataset_item in dataset_dir:
        item_path = os.path.join(base_dir, dataset_item)
        logger.debug("reading dataset %s", dataset_item)
        if not os.path.exists(os.path.join(item_path, 'left_camera')):
            # print "not a real path ", item_path
            continue
        
        data_stream = DataStream(os.path.join(item_path, 'left_camera'))
        raw_dataset = open( os.path.join(base_dir, dataset_item, 'left_camera.txt') )

        dataset = [[], []]
        line = raw_dataset.readline()

        while line:
            line = line.split(" ")

            if len(line) < 3:
                continue
            
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
            # processed.display_image('small', 'original_image')
            # continue
            boxes = processed.get_bounding_box()

            y,x = processed.image.shape
            
            processed.get_offset_from_center_in_rectangle_space(1, (x/5, 11*y/12), (4*x/5, y))
            processed.get_offset_from_center_in_rectangle_space(2, (x/10, 10*y/12), (9*x/10, 11*y/12))
            processed.get_offset_from_center_in_rectangle_space(3, (x/20, 9*y/12), (19*x/20, 10*y/12))
            processed.get_offset_from_center_in_rectangle_space(3, (x/30, 8*y/12), (29*x/30, 9*y/12))

            processed.offset_mapped_image.display_image('large', 'processing_offset')


            processed.draw_line((x/2, 0), (x/2, y))
            
            if boxes:
                bx,by,bw,bh = processed.get_closest_box(np.array(boxes)[:,:4])
                x1, y1, x2, y2 = x/2, y, int(bx), int(by)
                # processed.draw_line((x1,y1),(x2,y2))

                line1 = ((x1,y1),(x2,y2))
                line2 = ((x/2, 0), (x/2, y))
                # print processed.angle_between_lines(line1, line2)

            processed.display_image('medium', 'processed_image')
            # CVTools(image).display_image('small', 'original_image_rgb')

            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break
            
            # time.sleep(0.3)
            print "********** end of iteration ***************"
        
        cv.destroyAllWindows()

detect_lines(0,0,0,0)