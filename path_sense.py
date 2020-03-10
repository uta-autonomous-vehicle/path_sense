import os
import constants
import numpy as np
from PIL import Image
from data_reader import DataStream
import tensorflow as tf
import cv2 as cv

from test import Follower
import math

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

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
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 800, 800)

    cv.putText(img, "showing image:" + name, (200,200), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255) )
    cv.imshow(name, img)    

def canny_processing(frame, thres1, thres2):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)    
    # ret, frame = cv.threshold(frame, 50, 100, 0)
    frame = cv.GaussianBlur(frame, (5, 5), 0) # Applies a 5x5 gaussian blur
    frame = cv.Canny(frame, thres1, thres2)

    # contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)

    # lines = cv.HoughLinesP(frame, 1, np.pi/180, 10, 30, 10)
    # frame2 = np.zeros(frame.shape)
    # if not lines is None:
    #     for line in lines:
    #         x1,y1,x2,y2 = line[0]
    #         cv.line(frame2,(x1,y1),(x2,y2),(255,255,255), 2)
    return frame
    
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
    

if __name__ == "__main__":
    number_of_samples = 5
    for i in range(1, number_of_samples):
        data_stream = DataStream('local/data/{}/left_camera/'.format(i))
        dataset = [[], []]
        raw_dataset = open("local/data/{}/left_camera.txt".format(i))
        i = 0
        line = raw_dataset.readline()

        while line:
            i += 1
            # if i > 1000: break
            line = line.split(" ")
            
            image_name = line[0]
            steering = float(line[1])
            speed = float(line[2])
            
            # dataset[0].append(data_stream.get_item(image_name))
            # dataset[1].append(steering)
            line = raw_dataset.readline()

            image = data_stream.get_item(image_name)
            
            image2 = np.zeros((720, 1280))

            min_height = 3*IMAGE_HEIGHT/4 
            max_height = min_height + 100

            min_width = IMAGE_WIDTH/5 
            max_width = min_width + 500

            # creating a mask in the lower 3/4th region
            # image2[min_height:max_height,:] = image[min_height:max_height,:]
            image2[min_height:max_height,min_width:max_width] = canny_processing(image[min_height:max_height,min_width:max_width], 30, 90)
            # image2 = canny_processing(image, 30, 60)

            # center point
            draw_line(image2, IMAGE_WIDTH/2, min_height, IMAGE_WIDTH/2, max_height)
            
            # Calculates the max of (max of columns), indicating the direction where the segmentation was detected.
            # Not a good solution when we have other objects besides the track.
            maxes = np.argmax(image2[min_height:max_height,:], axis = 0)
            maxes = np.argmax(maxes)

            draw_line(image2, IMAGE_WIDTH/2, max_height, maxes, min_height)

            show_image("2", image2)

            # Follower().image_callback(image)

            if cv.waitKey(1) == ord('q'):
                break

        print "***"
        # print(len(dataset[0]))
        # dataset[0] = np.asarray(dataset[0])
        # dataset[1] = np.asarray(dataset[1])
        # lane_detector = LaneDetector(dataset)
        # break

