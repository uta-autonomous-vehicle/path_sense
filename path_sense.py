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

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'

cv.namedWindow(window_capture_name, cv.WINDOW_NORMAL)
cv.namedWindow(window_detection_name, cv.WINDOW_NORMAL)

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv/49590801
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def draw_line(img, x1, y1, x2, y2):
    cv.line(img,(x1,y1),(x2,y2),(255,255,255), 2)
    return

def draw_circle_on_frame(img, x,y):
    cv.circle(img, (x,y), 100, (0,0,0), -1)

def write_on_image(img, text = ""):
    cv.putText(img, text, (200,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255) )

def show_image(name, img):
    # print "showing image ", name
    # cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 600, 300)

    cv.putText(img, "showing image:" + name, (200,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255) )
    cv.imshow(name, img)

def canny_processing(frame, thres1, thres2, thresh3, thresh4):
    # print "canny proessing"
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)    
    # ret, frame = cv.threshold(frame, 50, 100, 0)
    frame = cv.GaussianBlur(frame, (5, 5), 0) # Applies a 5x5 gaussian blur
    # frame = cv.Canny(frame, thres1, thres2)
    frame = cv.Canny(frame, thres1, thres2)
    frame = cv.Canny(frame, thres1, thres2 + 10)
    # ret, frame = cv.threshold(frame, thresh3, thresh4, 0)
    frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)

    # frame = cv.connectedComponentsWithStats(frame, 4, cv.CV_32S)

    
    # contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)

    # lines = cv.HoughLinesP(frame, 1, np.pi/180, 10, 30, 10)
    # frame2 = np.zeros(frame.shape)
    # if not lines is None:
    #     for line in lines:
    #         x1,y1,x2,y2 = line[0]
    #         cv.line(frame2,(x1,y1),(x2,y2),(255,255,255), 2)
    return frame

def get_frame_for_ordinates(frame, x_offset, y_offset, window_size_width, window_size_height):
    if window_size_width and window_size_height:
        window_size_width = window_size_width/2
        window_size_height = window_size_height/2
    
    if x_offset:
        x_center = frame.shape[1]/2 + x_offset
        x_min, x_max = x_center - window_size_width, x_center + window_size_width
    else:
        x_min, x_max = 0, frame.shape[0]
    
    if y_offset:
        y_center = frame.shape[0]/2 + y_offset
        y_min, y_max = y_center - window_size_height, y_center + window_size_height
    else:
        y_min, y_max = 0, frame.shape[1]
    
    print "frame shape ", frame.shape
    print "xcenter, ycenter ", x_center, y_center
    print "xmin, xmax ", x_min, x_max
    print "ymin, ymax ", y_min, y_max

    frame = frame[y_min:y_max, x_min: x_max]
    return frame


def detect_box_in_frame(frame):
    maxes = np.argmax(frame, axis = 0)
    maxes = maxes.reshape((maxes.shape[0]/50, -1))

    maxes[maxes  > 0] = 1
    maxes = maxes.sum(axis = 1)
    maxes_sorted = maxes.argsort()[-3:][-1]

    # pdb.set_trace()

    # returns a percent value where it found the large object
    return (maxes_sorted * 50) / (frame.shape[1]*0.1)


def get_bounding_boxes_for_objects(frame):
    contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = {}
    highest_x = 0
    highest_y = 0
    highest_w = 0
    highest_h = 0
    for c in contours:
        area = cv.contourArea(c)
        print "area ", area
        # areas[c] = area
        x, y, w, h = cv.boundingRect(c)
        if highest_w < w and highest_h < h:
            highest_w = w
            highest_h = h
            highest_x = x
            highest_y = y
        # cv.drawContours(frame, c, -1, (255, 255, 255), 3)
        # cv.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 2)
    
    cv.rectangle(frame, (highest_x,highest_y), (highest_x+highest_w, highest_y+highest_h), (255, 255, 255), 2)
    return (highest_x, highest_y), frame

def filter_by_color(frame):
    low_hsv = (20, 150, 150)
    high_hsv = (30, 255, 255)
    # high_hsv = (30, 255, 255)
    print "filtering yellow ", low_hsv, high_hsv
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    filtered_frame = cv.inRange(frame, low_hsv, high_hsv )

    return filtered_frame

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
    

def show_canny_with_thres1(thres1, thres2, thresh3, thresh4):
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

            # creating a mask in the lower 3/4th region
            # image2[min_height:max_height,:] = image[min_height:max_height,:]
            # if not image:
            #     print "image is none"
            #     continue
            image2 = np.zeros((720, 1280))

            frame_new = filter_by_color(image)

            # frame_new = get_frame_for_ordinates(image, -320, -155, 640, 310)
            # frame_new = image[min_height:max_height, min_width:max_width]
            # image2[min_height:max_height,min_width:max_width] = canny_processing(image[min_height:max_height,min_width:max_width], thres1, thres2, thresh3, thresh4)
            # frame_new = canny_processing(frame_new, thres1, thres2, thresh3, thresh4)
            # # image2 = canny_processing(image, thres1, thres2, thresh3, thresh4)

            
            # # Calculates the max of (max of columns), indicating the direction where the segmentation was detected.
            # # Not a good solution when we have other objects besides the track.
            # # maxes = np.argmax(image2[min_height:max_height,:], axis = 0)
            # # maxes = np.argmax(maxes)
            
            # p_value = detect_box_in_frame(frame_new)
            (x1, y1), frame_new = get_bounding_boxes_for_objects(frame_new)
            # x_center_of_target_in_new_frame = int((frame_new.shape[1] * p_value))
            # print "p_value {} x_center_of_target_in_new_frame ".format(p_value)
            # import pdb; pdb.set_trace()
            draw_line(frame_new, frame_new.shape[1]/2, frame_new.shape[0], x1, y1)
            # draw_line(image, image.shape[1]/2, image.shape[0], p_value , 0)

            # # center point
            draw_line(frame_new, frame_new.shape[1]/2, 0, frame_new.shape[1]/2, frame_new.shape[1])
            # draw_line(image, image.shape[1]/2, 0, image.shape[1]/2, image.shape[1])

            show_image(window_capture_name, image)
            show_image(window_detection_name, frame_new)

            # Follower().image_callback(image)

            if cv.waitKey(1) == ord('q'):
                break
            if cv.waitKey(1) == ord('w'):
                time.sleep(1000.0)
            

        print "***"
        # print(len(dataset[0]))
        # dataset[0] = np.asarray(dataset[0])
        # dataset[1] = np.asarray(dataset[1])
        # lane_detector = LaneDetector(dataset)
        # break


for i in range(90,150, 20):
    for j in range(100,200,20):
        for k in range(10,100, 5):
            for l in range(50,200,5):
                # print("thesh ", i, j)
                print("canny ", k, l)
                show_canny_with_thres1(k,l, i,j )

first_image = os.listdir(os.path.join(os.listdir(os.path.abspath(constants.IMAGE_DIR))[0], "left_camera"))
im = cv.imread()