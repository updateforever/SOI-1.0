# import the necessary packages
import os

from .basetracker import Tracker
# from pytracking.tracker.base.basetracker import BaseTracker
from pytracking.evaluation.tracker import Tracker as PyTracker


class TrackerToMP(Tracker):
    def __init__(self):
        super(TrackerToMP, self).__init__(name='tomp', is_deterministic=True)
        params = PyTracker('tomp', 'tomp50').get_parameters()  # 获取相关先验参数
        self.tracker = PyTracker('tomp', 'tomp50').create_tracker(params)
        self.tracker_param = 'tomp50'

    def init(self, image, box):
        # print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']

        return out
