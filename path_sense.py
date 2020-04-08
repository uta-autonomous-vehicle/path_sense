# from __future__ import division
import time
import os
import constants
import numpy as np
from PIL import Image
from data_reader import DataStream
# import tensorflow as tf
import cv2 as cv

import pdb
from test import Follower
import math

from utils import CVTools, StraightLineOffsetDetector
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

def test_steering_angle_for_image(image):
    min_height = 720/3
    max_height = 720

    min_width = 0
    max_width = 1280/2

    image2 = np.zeros((720, 1280))

    processed = CVTools(image.copy())
    processed.enable_image_display()
    # NOTE: http://www.flatuicolorpicker.com/yellow-hsv-color-model
    processed.filter_color((58, 30, 30), (64, 100, 100))
    # processed.display_image('original_image')
    # continue
    boxes = processed.get_bounding_box(True)

    y,x = processed.image.shape

    sampling = 50
    offset = 0
    weighted_offset = 0
    line_iter = 0
    for i in range(1, sampling):
        x_partitions = 5*i
        y_partitions = 3*sampling

        y_range = y_partitions - i

        x_from = x/x_partitions
        x_to = x * (x_partitions - 1)/x_partitions
        y_from = y * (y_range)/y_partitions
        y_to = y * (y_range + 1)/y_partitions

        offset_iter = processed.get_offset_from_center_in_rectangle_space((x_from, y_from), (x_to, y_to))
        offset += offset_iter
        weighted_offset += (0.1 * i) * offset_iter

    avg_offset = offset/(1.0 * (sampling-1))

    p_value_offset = avg_offset / (x/2)
    steering_offset = 0.34 * p_value_offset

    # weighted_avg_offset = weighted_offset/(1.0 * (sampling-1))

    logger.info("average offset for test space: %s", avg_offset)
    logger.info("average offset for test space: %s%%", p_value_offset)
    logger.info("steering offset for test space: %s%%", steering_offset)
    
    straight_line_offset_detector = StraightLineOffsetDetector(image)

    straight_line_offset_detector_offset = straight_line_offset_detector.get_steering_angle()
    logger.info("test for StraightLineOffsetDetector: %s ", straight_line_offset_detector_offset-steering_offset)
    # logger.info("average weighted offset for this space: %s", weighted_avg_offset)

    processed.offset_mapped_image.image[:7*y/12,:] = [0,0,0]
    processed.draw_line((x/2, 0), (x/2, y))

    processed.draw_circle((int(x/2 + avg_offset), y/2), color=constants.WHITE)
    
    avg_offset_coordinates = (int(x/2 - avg_offset), 7*y/8)
    # weighted_avg_offset_coordinates = (int(x/2 - weighted_avg_offset), 7*y/8)
    
    processed.offset_mapped_image.draw_circle(avg_offset_coordinates, color=constants.RED)
    # processed.offset_mapped_image.draw_circle(weighted_avg_offset_coordinates, color=constants.GREEN)
    # processed.offset_mapped_image.draw_line((x/2,y), avg_offset_coordinates)
    processed.offset_mapped_image.enable_image_display()
    processed.offset_mapped_image.display_image('processing_offset')

    if boxes:
        bx,by,bw,bh = processed.get_closest_box(np.array(boxes)[:,:4])
        x1, y1, x2, y2 = x/2, y, int(bx), int(by)
        # processed.draw_line((x1,y1),(x2,y2))

        line1 = ((x1,y1),(x2,y2))
        line2 = ((x/2, 0), (x/2, y))
        # print processed.angle_between_lines(line1, line2)
    
    processed.display_image('processed_image')
    CVTools(image).display_image('original_image_rgb')


def detect_lines(data_stream, dataset):
    for data in dataset:
        image_name, steering, speed = data
        image = data_stream.get_item(image_name)
            # im = cv.imread(image_name)
        # cv.imshow("original image", image)
        # pdb.set_trace()
        
        # logger.debug("image size %s", image.shape)
        tool = CVTools(image)
        # tool.display_image('test')
        
        detector = StraightLineOffsetDetector(image, True)
        
        # detector.filter_color("yellow")
        detector.filter_color("pink")

        logger.debug("%s steering", detector.get_steering_angle())
        
        cv.imshow("2", detector.image.image)
        # cv.imshow("3", detector.image.offset_mapped_image.image)

        if cv.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    number_of_samples = 5
    base_dir = os.path.abspath(constants.IMAGE_DIR)
    dataset_dir = os.listdir(base_dir)

    logger.debug("number of datasets %s", len(dataset_dir))

    for datatset_iter, dataset_item in enumerate(dataset_dir):
        # time.sleep(1)
        item_path = os.path.join(base_dir, dataset_item)
        logger.debug("reading dataset %s", dataset_item)

        left_exists = os.path.exists(os.path.join(item_path, 'left_camera'))
        right_exists = os.path.exists(os.path.join(item_path, 'right_camera'))
        
        if left_exists:
            data_stream = DataStream(os.path.join(item_path, 'left_camera'))
            list_of_images = os.listdir(os.path.join(item_path, 'left_camera'))
        elif right_exists:
            data_stream = DataStream(os.path.join(item_path, 'right_camera'))
            list_of_images = os.listdir(os.path.join(item_path, 'right_camera'))
        else:
            continue

        dataset = []
        line_iter = 0
        
        logger.debug("reading dataset %s with %s items", dataset_item, len(list_of_images))
        for image_index in range( len(list_of_images) + 100):
            # logger.info("reading image: %s", image_index)
            image_name = '{}.jpg'.format(image_index)
            if image_name in list_of_images:
                # image = data_stream.get_item(image_name)
                dataset.append((image_name, 0, 0))

        logger.debug("read {} images".format(len(dataset)))
        detect_lines(data_stream, dataset)


cv.destroyAllWindows()