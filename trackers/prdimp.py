# import the necessary packages
import os

from uav.trackers import Tracker
# from pytracking.tracker.base.basetracker import BaseTracker
from pytracking.evaluation.tracker import Tracker as PyTracker


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


class TrackerPrDiMP(Tracker):
    def __init__(self):
        super(TrackerPrDiMP, self).__init__(name='PrDiMP', is_deterministic=True)
        params = PyTracker('dimp', 'prdimp50').get_parameters()  # 获取相关先验参数
        self.tracker = PyTracker('dimp', 'prdimp50').create_tracker(params)

    def init(self, image, box):
        print(box)
        self.tracker.initialize(image, {'init_bbox': box})

    def update(self, image):
        out = self.tracker.track(image)
        out = out['target_bbox']
        return out
