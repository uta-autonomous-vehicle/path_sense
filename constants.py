# NOTE: copy datasets to dir IMAGE_DIR. naming is not important
import os

BASE_DIR = os.path.join(os.getcwd(), "../")

IMAGE_DIR = os.path.join(BASE_DIR, 'datasets', 'dataset_detect_stopping_points')
VIDEO_DIR = 'asset/videos'

STREAM_TYPE_IMAGES = 'images'
STREAM_TYPE_VIDEOS = 'videos'


# RBG vs BGR

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (255, 255, 255)