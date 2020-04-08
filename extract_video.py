import os
import cv2 as cv
import numpy as np
import constants

def convert_images_to_video_seq(base_path):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    number_of_items = os.listdir(os.path.join(base_path, 'left_camera'))
    if number_of_items:
        left_camera_video = cv.VideoWriter(base_path + '/left_camera.mp4', fourcc, 15,(1280,720))
        for i in range(len(number_of_items)):
            path_name = base_path + '/left_camera/{}.jpg'.format(i)
            if os.path.exists(os.path.abspath(path_name)):
                left_camera_video.write(cv.imread(path_name))
        left_camera_video.release()

    number_of_items = os.listdir(os.path.join(base_path, 'right_camera'))
    if number_of_items:
        right_camera_video = cv.VideoWriter(base_path + '/right_camera.mp4', fourcc, 15,(1280,720))
        for i in range(len(number_of_items)):
            path_name = base_path + '/right_camera/{}.jpg'.format(i)
            if os.path.exists(os.path.abspath(path_name)):
                right_camera_video.write(cv.imread(path_name))
        right_camera_video.release()


current_path = os.getcwd()
base_dir = os.path.abspath(constants.IMAGE_DIR)
dataset_list = os.listdir(base_dir)

for ds in dataset_list:
    datatset_dir = os.path.join(base_dir, ds)
    if not os.path.isdir(datatset_dir):
        print datatset_dir, " is not a directory"
        continue

    left_camera_dir = os.path.join(datatset_dir, 'left_camera')
    right_camera_dir = os.path.join(datatset_dir, 'right_camera')
    if os.path.exists(left_camera_dir) and os.path.exists(right_camera_dir):
        print "processing ", left_camera_dir, " and ", right_camera_dir
        convert_images_to_video_seq(datatset_dir)
