# import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
import argparse
# import imutils
import time
import cv2 as cv
import os

from uav.trackers import Tracker
# from pytracking.tracker.base.basetracker import BaseTracker
from pytracking.evaluation.tracker import Tracker as PyTracker
import importlib


def get_filename(folderpath):
    filenames = []
    for root, dirs, files in os.walk(folderpath):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filenames.append(os.path.join(root, file))
    filenames.sort(key=lambda x: int(x[-10:-4]))
    return filenames


def makedir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        return True
    else:
        return False


class TrackerDiMP(Tracker):
    def __init__(self):
        super(TrackerDiMP, self).__init__(name='DiMP', is_deterministic=True)

        params = PyTracker('dimp', 'dimp50').get_parameters()  # 获取相关先验参数
        self.tracker = PyTracker('dimp', 'dimp50').create_tracker(params)

    def init(self, image, box):
        print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']
        return out
