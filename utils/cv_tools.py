import time
import math
import os
import numpy as np
from PIL import Image
from data_reader import DataStream
import tensorflow as tf
import cv2 as cv

import pdb
from test import Follower
import math


class CVTools(object):
    def __init__(self, image):
        self.image = image
        return
    
    def display_image(self, size = 'small', name = 'default'):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        if size == 'small':
            cv.resizeWindow(name, 400, 300)
        elif size == 'medium':
            cv.resizeWindow(name, 800, 600)
        elif size == 'large':
            cv.resizeWindow(name, 1200, 800)

        cv.imshow(name, self.image)
    
    def add_text_to_image(self, text, position = (0,0)):
        cv.putText(self.image, text, position, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255) )
    
    def draw_circle(self, position):
        cv.circle(self.image, position, 100, (255,255,255), -1)
    
    def draw_line(self, start, end):
        cv.line(self.image,start, end, (255,255,255), 2)
    
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
    
    def filter_color(self, low, high):
        # NOTE: color space is different for opencv  - [(0-179), (0-255), (0-255)] vs [(0-360), (0-100), (0-100)]
        multiplier_hsv_for_opencv = [0.5, 2.55, 2.55]
        low_hsv = np.multiply(low, multiplier_hsv_for_opencv)
        high_hsv = np.multiply(high, multiplier_hsv_for_opencv)

        self.reduce_to_hsv()
        self.image = cv.inRange(self.image, low_hsv, high_hsv )

    def get_contours(self):
        # TODO: try other modes
        contours, hierarchy = cv.findContours(self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def get_bounding_box(self, draw = True):
        contours = self.get_contours()
        areas = []
        shape = self.image.shape

        # NOTE: new_image displays in RBG
        new_image = np.zeros((shape[0], shape[1], 3), dtype = np.uint8)
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            # hull = cv.convexHull(c)
            x, y, w, h = cv.boundingRect(c)
            areas.append([x, y, w, h, area])

            if draw:
                cv.rectangle(self.image, (x,y), (x+w, y+h), (255, 255, 0), 1)
                cv.rectangle(new_image, (x,y), (x+w, y+h), (0, 0, 200), 1)
                # cv.drawContours(self.image, hull, -1, (255, 255, 255), 1)
                if area > 10 and area < 100:
                    self.add_text_to_image("area:{}".format(area), (x,y))
                if w > 30:
                    # NOTE: showing nearest and widest detected shapes
                    cv.rectangle(new_image, (x,y), (x+w, y+h), (255, 0, 0), 1)

                    x_offset = ((2*x) + w)/2 - 50
                    y_offset = ((2*y) + h)/2 - 50
                    if y > self.image.shape[0]/2 and w > 50:
                        cv.rectangle(new_image, (x,y), (x+50, y+50), (255, 255, 0), 1)
        
        new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
        CVTools(new_image).display_image('medium', 'processing_boxing')
        return areas

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
