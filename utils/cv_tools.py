import time
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
        self.image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
    
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
        low_hsv = low
        high_hsv = high
        
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
        new_image = np.zeros((shape[0], shape[1]), dtype = np.uint8)
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            hull = cv.convexHull(c)
            x, y, w, h = cv.boundingRect(c)
            areas.append(((x,y,x+w,y+h), area))

            if draw:
                if area > 10 and area < 100:
                    self.add_text_to_image("area:{}".format(area), (x,y))
                cv.rectangle(self.image, (x,y), (x+w, y+h), (255, 255, 0), 2)
                # cv.drawContours(self.image, hull, -1, (255, 255, 255), 1)
        
        return areas


