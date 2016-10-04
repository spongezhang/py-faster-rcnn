# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from fast_rcnn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
      return [PIL.Image.open(self.image_path_at(i)).size[0]
              for i in xrange(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2


    def create_roidb_from_box_list(self, box_list_1, box_list_2, gt_roidb):
        assert len(box_list_1) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        assert len(box_list_2) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes_1 = box_list_1[i]
            num_boxes_1 = boxes_1.shape[0]
            overlaps_1 = np.zeros((num_boxes_1, self.num_classes), dtype=np.float32)

            if gt_roidb_1 is not None and gt_roidb[i]['boxes_1'].size > 0:
                gt_boxes_1 = gt_roidb[i]['boxes_1']
                gt_classes_1 = gt_roidb[i]['gt_classes_1']
                gt_overlaps_1 = bbox_overlaps(boxes_1.astype(np.float),
                                            gt_boxes_1.astype(np.float))
                argmaxes_1 = gt_overlaps_1.argmax(axis=1)
                maxes_1 = gt_overlaps_1.max(axis=1)
                I = np.where(maxes_1 > 0)[0]
                overlaps_1[I, gt_classes_1[argmaxes_1[I]]] = maxes_1[I]

            overlaps_1 = scipy.sparse.csr_matrix(overlaps_1)
            
            boxes_2 = box_list_2[i]
            num_boxes_2 = boxes_2.shape[0]
            overlaps_2 = np.zeros((num_boxes_2, self.num_classes), dtype=np.float32)

            if gt_roidb_2 is not None and gt_roidb[i]['boxes_2'].size > 0:
                gt_boxes_2 = gt_roidb[i]['boxes_2']
                gt_classes_2 = gt_roidb[i]['gt_classes_2']
                gt_overlaps_2 = bbox_overlaps(boxes_2.astype(np.float),
                                            gt_boxes_2.astype(np.float))
                argmaxes_2 = gt_overlaps_2.argmax(axis=1)
                maxes_2 = gt_overlaps_2.max(axis=1)
                I = np.where(maxes_2 > 0)[0]
                overlaps_2[I, gt_classes_2[argmaxes[I]]] = maxes_2[I]

            overlaps_2 = scipy.sparse.csr_matrix(overlaps_2)
            roidb.append({
                'boxes_1' : boxes_1,
                'boxes_2' : boxes_2,
                'gt_classes_1' : np.zeros((num_boxes_1,), dtype=np.int32),
                'gt_classes_2' : np.zeros((num_boxes_2,), dtype=np.int32),
                'gt_overlaps_1' : overlaps_1,
                'gt_overlaps_2' : overlaps_2,
                'flipped_1' : False,
                'flipped_2' : False,
                'seg_areas_1' : np.zeros((num_boxes_1,), dtype=np.float32),
                'seg_areas_2' : np.zeros((num_boxes_2,), dtype=np.float32),
            })
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes_1'] = np.vstack((a[i]['boxes_1'], b[i]['boxes_1']))
            a[i]['gt_classes_1'] = np.hstack((a[i]['gt_classes_1'],
                                            b[i]['gt_classes_1']))
            a[i]['gt_overlaps_1'] = scipy.sparse.vstack([a[i]['gt_overlaps_1'],
                                                       b[i]['gt_overlaps_1']])
            a[i]['seg_areas_1'] = np.hstack((a[i]['seg_areas_1'],
                                           b[i]['seg_areas_1']))
            a[i]['boxes_2'] = np.vstack((a[i]['boxes_2'], b[i]['boxes_2']))
            a[i]['gt_classes_2'] = np.hstack((a[i]['gt_classes_2'],
                                            b[i]['gt_classes_2']))
            a[i]['gt_overlaps_2'] = scipy.sparse.vstack([a[i]['gt_overlaps_2'],
                                                       b[i]['gt_overlaps_2']])
            a[i]['seg_areas_2'] = np.hstack((a[i]['seg_areas_2'],
                                           b[i]['seg_areas_2']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
