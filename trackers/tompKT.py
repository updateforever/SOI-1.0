# import the necessary packages
import os

from .basetracker import Tracker
# from pytracking.tracker.base.basetracker import BaseTracker
from pytracking.evaluation.tracker import Tracker as PyTracker

class TrackerToMPKT(Tracker):
    def __init__(self):
        super(TrackerToMPKT, self).__init__(name='tompKT', is_deterministic=True)
        params = PyTracker('tompKT', 'default').get_parameters()  # 获取相关先验参数
        self.tracker = PyTracker('tompKT', 'default').create_tracker(params)
        self.tracker_param = 'default'

    def init(self, image, box):
        # print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']

        return out
