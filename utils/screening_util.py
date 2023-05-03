import json
import numpy as np
import torch
import math
import cv2 as cv
import random
import torch.nn.functional as F
from .tensor import TensorList
import os


def find_local_maxima(scores, th, ks):
    """Find local maxima in a heat map.
        args:
            scores - heat map to find the local maxima in.
            th - threshold that defines the minamal value needed to be considered as a local maximum.
            ks = local neighbourhood (kernel size) specifiying the minimal distance between two maxima.

        returns:
            coordinates and values of the local maxima.
    """
    ndims = scores.ndim

    if ndims == 2:
        scores = scores.view(1, 1, scores.shape[0], scores.shape[1])

    scores_max = F.max_pool2d(scores, kernel_size=ks, stride=1, padding=ks // 2)  # 池化操作找极大值

    peak_mask = (scores == scores_max) & (scores > th)  # 找出位置
    coords = torch.nonzero(peak_mask)
    intensities = scores[peak_mask]

    # Highest peak first
    idx_maxsort = torch.argsort(-intensities)
    coords = coords[idx_maxsort]
    intensities = intensities[idx_maxsort]

    if ndims == 4:

        coords_batch, intensities_batch, = TensorList(), TensorList()
        for i in range(scores.shape[0]):
            mask = (coords[:, 0] == i)
            coords_batch.append(coords[mask, 2:])
            intensities_batch.append(intensities[mask])
    else:
        coords_batch = coords[:, 2:]
        intensities_batch = intensities

    return coords_batch, intensities_batch


def load_dump_seq_data_from_disk(path):
    d = {}

    if os.path.exists(path):
        with open(path, 'r') as f:
            d = json.load(f)

    return d


def dump_seq_data_to_disk(save_path, seq_name, seq_data):
    d = load_dump_seq_data_from_disk(save_path)

    d[seq_name] = seq_data

    with open(save_path, 'w') as f:
        json.dump(d, f)


def determine_frame_state(tracking_data, tracker, seq, th=0.25):
    visible = seq.target_visible[tracker.frame_num - 1]
    num_candidates = tracking_data['target_candidate_scores'].shape[0]

    state = None
    if num_candidates >= 2:
        max_candidate_score = tracking_data['target_candidate_scores'].max()

        anno_and_target_candidate_score_dists = torch.sqrt(
            torch.sum((tracking_data['target_anno_coord'] - tracking_data['target_candidate_coords']) ** 2,
                      dim=1).float())

        ids = torch.argsort(anno_and_target_candidate_score_dists)

        score_dist_pred_anno = anno_and_target_candidate_score_dists[ids[0]]
        sortindex_correct_candidate = ids[0]
        score_dist_anno_2nd_highest_score_candidate = anno_and_target_candidate_score_dists[
            ids[1]] if num_candidates > 1 else 10000

        if (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
                sortindex_correct_candidate == 0 and max_candidate_score < th and visible != 0):
            state = 'G'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate == 0 and max_candidate_score >= th and visible != 0):
            state = 'H'
        elif (num_candidates > 1 and score_dist_pred_anno > 4 and max_candidate_score >= th and visible != 0):
            state = 'J'
        elif (num_candidates > 1 and score_dist_pred_anno <= 2 and score_dist_anno_2nd_highest_score_candidate > 4 and
              sortindex_correct_candidate > 0 and max_candidate_score >= th and visible != 0):
            state = 'K'

    return state


def determine_subseq_state(frame_state, frame_state_previous):
    if frame_state is not None and frame_state_previous is not None:
        return '{}{}'.format(frame_state_previous, frame_state)
    else:
        return None


def extract_candidate_data(data, th=0.1):
    tracker_data = data
    search_area_box = tracker_data['search_area_box']
    score_map = tracker_data['score_map'].cpu()

    # max_score = score_map.max()

    target_candidate_coords, target_candidate_scores = find_local_maxima(score_map.squeeze(), th=th,
                                                                         ks=5)  # 0.05  0.1  0.2  0.3  0.4
    tg_num = len(target_candidate_scores)

    if 'x_dict' in tracker_data.keys():
        search_img = tracker_data['x_dict'].tensors
        return dict(search_area_box=search_area_box,
                    target_candidate_scores=target_candidate_scores,
                    target_candidate_coords=target_candidate_coords,
                    search_img=search_img, ), \
               tg_num
    else:
        return dict(search_area_box=search_area_box,
                    target_candidate_scores=target_candidate_scores,
                    target_candidate_coords=target_candidate_coords, ), \
               tg_num


def update_seq_data(seq_candidate_data, frame_candidate_data):
    for key, val in frame_candidate_data.items():
        val = val.float().tolist() if torch.is_tensor(val) else val
        seq_candidate_data[key].append(val)


def extract_score_map(data):
    tracker_data = data
    score_map = tracker_data['score_map'].cpu()
    score_map = score_map.flatten()

    return dict(sm=score_map)


def extract_candidate_set(data, anno, th=0.4, screen_mode='MaxPooling', alpha=0.8, save_result=False):
    tracker_data = data  # data ==> {'sm':{'num':{'score_map'}}}
    assert len(data['score_map_list']) + 1 == len(anno)

    score_map = tracker_data['score_map_list']  # {'num':{'score_map'}}
    seq_candidate_scores = []
    seq_candidate_coords = []
    seq_tc_num = []
    for i, v in enumerate(score_map):
        frame_anno = anno[i+1]
        sz = int(math.sqrt(len(v)))
        flatten_score_map = np.array(v)
        re_score_map = torch.tensor(flatten_score_map.reshape(sz, sz))
        if screen_mode == 'MaxPooling':
            target_candidate_coords, target_candidate_scores = find_local_maxima(re_score_map.squeeze(), th=th,
                                                                                 ks=5)  # 0.05  0.1  0.2  0.3  0.4
            tc_num = len(target_candidate_scores)
        elif screen_mode == 'MAX_SCORE':
            target_candidate_coords, target_candidate_scores = find_local_maxima(re_score_map.squeeze(), th=alpha,
                                                                                 ks=5)  # 0.05  0.1  0.2  0.3  0.4
            tc_num = len(target_candidate_scores)
        else:
            raise Exception('screen_mode error.')
        if save_result:
            seq_candidate_scores.append(target_candidate_scores)
            seq_candidate_coords.append(target_candidate_coords)
        seq_tc_num.append(tc_num)
        # if frame_anno == [0., 0., 0., 0.] and tc_num > 0:
        #     return

    return seq_candidate_scores, seq_candidate_coords, seq_tc_num
