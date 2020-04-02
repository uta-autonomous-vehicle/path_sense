# from __future__ import division
import time
import math
import os
import numpy as np
from PIL import Image
# from data_reader import DataStream
# import tensorflow as tf
import cv2 as cv

import pdb
# from test import Follower
import math

from logger import logger

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (255, 255, 255)

YELLOW_COLOR_RANGE = ((58, 30, 30), (64, 100, 100))

class CVTools(object):
    """
        All processing is stored locally when using an instance of this object. to self.image
        No methods return self, do hierarchical calling isn't possible.

    """
    def __init__(self, image, tool_instance_name = 'default'):
        self.original_image = image
        self.image = image
        self.tool_instance_name = tool_instance_name
        self.image_shape = image.shape

        self.offset_mapped_image = None

        self.set_tracking_values()
    
    def disable_image_display(self):
        self.image_can_display = False

    def enable_image_display(self):
        self.image_can_display = True
    
    def set_tracking_values(self):
        self.image_is_color_filtered = False
        self.image_is_binary = False
        self.image_is_processed_for_edges = False
        self.image_is_displayed = False
        self.image_can_display = False

    def display_image(self, name = 'default'):
        if self.image_can_display:
            cv.namedWindow(self.tool_instance_name + name, cv.WINDOW_NORMAL)
            cv.imshow(self.tool_instance_name + name, self.image)
    
    def resize_window(self, name = 'default', size = 'small'):
        if size == 'small':
            cv.resizeWindow(name, 400, 300)
        elif size == 'medium':
            cv.resizeWindow(name, 800, 600)
        elif size == 'large':
            cv.resizeWindow(name, 1200, 800)
    
    def add_text_to_image(self, text, position = (0,0)):
        cv.putText(self.image, text, position, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255) )
    
    def draw_circle(self, position, radius = 10, color = WHITE):
        cv.circle(self.image, position, radius, color, -1)
    
    def draw_rectangle(self, p1, p2):
        cv.rectangle(self.image, p1, p2, (255,255,255), 2)
    
    def draw_line(self, start, end, color = (255, 255, 255)):
        cv.line(self.image,start, end, color, 2)
    
    def reduce_to_gray(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    
    def reduce_to_hsv(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)

    def process_threshold(self, thresh1, thresh2):
        self.image = cv.GaussianBlur(self.image, (5, 5), 0)

    def process_canny(self, thresh1, thresh2, canny_thresh1, canny_thresh2):
        # print "canny proessing"
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        # ret, frame = cv.threshold(frame, 50, 100, 0)
        self.reduce_to_gray()
        frame = cv.GaussianBlur(self.image, (5, 5), 0) # Applies a 5x5 gaussian blur
        frame = cv.Canny(frame, thresh1, thresh2)
        frame = cv.Canny(frame, thresh1, thresh2 + 10)
        # ret, frame = cv.threshold(frame, thresh1, thresh2, 0)
        self.image = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    
    def detect_boxes(self):
        # FIXME: modify to adapt here.
        # AUTHOR: HP
        maxes = np.argmax(self.image, axis = 0)
        maxes = maxes.reshape((maxes.shape[0]/50, -1))

        maxes[maxes  > 0] = 1
        maxes = maxes.sum(axis = 1)
        maxes_sorted = maxes.argsort()[-3:][-1]

        # returns a percent value where it found the large object
        return (maxes_sorted * 50) / (self.image.shape[1]*0.1)
    
    def filter_color(self, low = YELLOW_COLOR_RANGE[0], high = YELLOW_COLOR_RANGE[1]):
        # NOTE: color space is different for opencv  - [(0-179), (0-255), (0-255)] vs [(0-360), (0-100), (0-100)]
        multiplier_hsv_for_opencv = [0.5, 2.55, 2.55]
        low_hsv = np.multiply(low, multiplier_hsv_for_opencv)
        high_hsv = np.multiply(high, multiplier_hsv_for_opencv)

        self.reduce_to_hsv()
        self.image = cv.inRange(self.image, low_hsv, high_hsv )

        self.image_is_color_filtered = True

    def get_contours(self):
        # TODO: try other modes
        _, contours, hierarchy = cv.findContours(self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours
    
    def detect_lines(self):
        return
    
    def get_offset_from_center_in_rectangle_space(self, p1 = (0,0), p2 = (0,01)):
        if not self.image_is_color_filtered:
            logger.error("image is not color filtered before requesting offset calculation")
            return 0

        y,x = self.image.shape
        self.offset_mapped_image = self.offset_mapped_image or CVTools(self.original_image)

        # x1, x2, y1, y2 = x/6, 5*x/6, 3*y/4, y
        x1, x2, y1, y2 = p1[0], p2[0], p1[1], p2[1]
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        y_center = (y1+y2)/2
        x_center = (x1+x2)/2
        region_of_interest = self.image[y1:y2,x1:x2]
        offset = 0

        if region_of_interest.any():
            y_focused_width, x_focused_width = region_of_interest.shape
            x_detected = region_of_interest.sum(axis=0).argsort()[-1]

            offset = x_focused_width/2 - x_detected
            # logger.info("offset for space x=(%s, %s) y=(%s, %s): %s", x1, x2, y1, y2, offset)

            # self.draw_line((x/2, y_center), (int(x/2 - offset), y_center))

            self.offset_mapped_image.add_text_to_image("offset: {}".format(offset), (x_center, y_center))
            self.offset_mapped_image.draw_line((x/2, y_center), (int(x/2 - offset), y_center), color=WHITE)
            
            self.offset_mapped_image.draw_rectangle((x1, y1), (x2, y2))
            self.offset_mapped_image.draw_line((x/2, y1), (x/2, y2))

        return offset

    def show_correct_path(self):
        return

    def get_bounding_box(self, draw = False):
        contours = self.get_contours()
        boxes = []
        largest_box_area = 0
        shape = self.image.shape

        # NOTE: new_image displays in RBG
        new_image = np.zeros((shape[0], shape[1], 3), dtype = np.uint8)
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            hull = cv.convexHull(c)
            x, y, w, h = cv.boundingRect(c)
            boxes.append([x, y, w, h, area])

            if area > largest_box_area: largest_box_area = area

            if draw:
                # cv.rectangle(self.image, (x,y), (x+w, y+h), (255, 255, 0), 1)
                cv.rectangle(new_image, (x,y), (x+w, y+h), (255, 0, 0), 1)

                # cv.drawContours(self.image, hull, -1, (255, 255, 255), 1)
                cv.drawContours(new_image, hull, -1, (255, 0, 0), 1)

                if area > 10 and area < 100:
                    self.add_text_to_image("area:{}".format(area), (x,y))
                    pass
                if w > 30:
                    # NOTE: showing nearest and widest detected shapes
                    cv.rectangle(new_image, (x,y), (x+w, y+h), (255, 0, 0), 1)

                    x_offset = ((2*x) + w)/2 - 50
                    y_offset = ((2*y) + h)/2 - 50
                  
                    if y > self.image.shape[0]/2 and w > 50:
                        cv.rectangle(new_image, (x,y), (x+50, y+50), (255, 255, 0), 1)
        
        if draw:
            logger.info("larges contour area found at: %s", largest_box_area)
            new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
            CVTools(new_image).display_image('processing_boxing')
        return boxes

    def get_closest_box(self, boxes):
        # TODO: finish implementation. return closest big box
        """ 
            image for opencv indexes (0,0) from top left

            boxes: list of boxes np array values:[ [x, y, w, h] ]
            rtype: (x, y, w, h)
        """

        sorted_by_w = boxes[(-boxes[:,3]).argsort()][:3]
        sorted_by_y = sorted_by_w[(-sorted_by_w[:,2]).argsort()]

        return sorted_by_y[0]
    
    def angle_between_lines(self, line1, line2):
        # TODO: finish implementation. return closest big box
        """
            return angle for the 

            line1: ((x,y), (x,y)) start and end co-ordinates for line
            line2: ((x,y), (x,y)) start and end co-ordinates for line
        """
        theta = 0
        
        return theta


class StraightLineOffsetDetector(object):
    def __init__(self, image):
        self.image = CVTools(image.copy(), '2')
        self.image.disable_image_display()

    def get_steering_angle(self):
        self.image.filter_color()
        # boxes = self.image.get_bounding_box()

        y,x = self.image.image.shape

        sampling = 50
        offset = 0
        weighted_offset = 0
        for current_sample in range(1, sampling):
            x_partitions = 5*current_sample
            y_partitions = 3*sampling

            y_range = y_partitions - current_sample

            x_from = x/x_partitions
            x_to = x * (x_partitions - 1)/x_partitions
            y_from = y * (y_range)/y_partitions
            y_to = y * (y_range + 1)/y_partitions

            offset_iter = self.image.get_offset_from_center_in_rectangle_space((x_from, y_from), (x_to, y_to))
            # print " offset_iter ", offset_iter
            offset += offset_iter
            weighted_offset += (0.1 * current_sample) * offset_iter

        avg_offset = offset/(1.0 * (sampling-1))
        p_value_offset = avg_offset / (x/4)
        steering_offset = 0.34 * p_value_offset

        # logger.info("average offset for detector space: %s", avg_offset)
        # logger.info("average offset for detector space: %s%%", p_value_offset)
        # logger.info("steering offset for detector space: %s%%", steering_offset)

        self.image.offset_mapped_image.display_image()

        return steering_offset
