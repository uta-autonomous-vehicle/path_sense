import os
import constants
from data_reader import DataStream

class LaneDetector(object):
    def __init__(self):
        # self.feed = ''
    
    def train(self, data_source):
        self.feed = DataStream(data_source)

    def test(self, data_source):
        self.feed = DataStream(data_source)

    def get_accuracy(self):
        return None
    
    def get_loss(self):
        return None