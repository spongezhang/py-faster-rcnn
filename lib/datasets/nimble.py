# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import siamese_imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class nimble(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'voc_' + '2016' + '_' + image_set)
        
        self._data_path = os.path.join('/Users/Xu/program/Image_Genealogy/data/NIMBLE_Hard')
        self._classes = ('__background__', 'copy')
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index_probe = self._load_image_set_index()
        self._image_index_world = self._load_image_set_index()
        
        # Default to roidb handler
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_2_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._data_path,'query',
                                  self.roidb[i].donor_file)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,'query',
                                  index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_nimble_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_proposal_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_proposal(self, index):
        filename = os.path.join(self._data_path, 'proposal', index + '.json')
        #currently only one bbox is considered.
        assert os.path.exists(cache_file),'Annotation {} has to be here'.format(filename)
        raw_data = sio.loadmat(filename)['boxes_1'].ravel()
        boxes_1 = raw_data
        keep = ds_utils.unique_boxes(boxes_1)
        boxes_1 = boxes_1[keep, :]
        keep = ds_utils.filter_small_boxes(boxes_2, self.config['min_size'])
        boxes_1 = boxes_1[keep, :]

        raw_data = sio.loadmat(filename)['boxes_2'].ravel()
        boxes_2 = raw_data
        keep = ds_utils.unique_boxes(boxes_2)
        boxes_2 = boxes_2[keep, :]
        keep = ds_utils.filter_small_boxes(boxes_2, self.config['min_size'])
        boxes_2 = boxes_2[keep, :]
            
        return boxes1, boxes_2


    def _load_selective_search_roidb(self, gt_roidb):
        
        box_list_1 = []
        box_list_2 = []
        for index in self._image_index:
            boxes_1, boxes_2 = self._load_proposal(index)
            box_list_1.append(boxes_1)
            box_list_2.append(boxes_2)
            
        return self.create_roidb_from_box_list(box_list_1,box_list_2, gt_roidb)

    def _load_nimble_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations_Python', index + '.json')
        #currently only one bbox is considered.
        assert os.path.exists(cache_file),'Annotation {} has to be here'.format(filename)
        
        num_objs = 1
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        f = open(filename,'r')

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = 1 
            gtboxes_1[ix, :] = obj.bbox
            gtboxes_2[ix,:] = obj.gtbbox
            gt_classes_1[ix] = cls
            overlaps_1[ix, cls] = 1.0
            seg_areas_1[ix] = 0
            gt_classes_1[ix] = cls
            overlaps_1[ix, cls] = 1.0
            seg_areas_1[ix] = 0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'donor_file' : donor_file,
                'boxes_1' : gtboxes_1,
                'boxes_2' : gtboxes_2,
                'gt_classes_1': gt_classes_1,
                'gt_overlaps_1' : overlaps_1,
                'gt_classes_2': gt_classes_2,
                'gt_overlaps_2' : overlaps_2,
                'flipped' : False,
                'seg_areas_1' : seg_areas_1,
                'seg_areas_2' : seg_areas_2}

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
