# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob_1, im_blob_2, im_scales_1, im_scales_2 = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data_1': im_blob_1}
    blobs = {'data_2': im_blob_2}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales_1) == 1, "Single batch only"
        assert len(roidb_1) == 1, "Single batch only"
        assert len(im_scales_2) == 1, "Single batch only"
        assert len(roidb_2) == 1, "Single batch only"

        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds_1 = np.where(roidb[0]['gt_classes_1'] != 0)[0]
        gt_boxes_1 = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes_1[:, 0:4] = roidb[0]['boxes_1'][gt_inds_1, :] * im_scales_1[0]
        gt_boxes_1[:, 4] = roidb[0]['gt_classes_1'][gt_inds_1]
        blobs['gt_boxes_1'] = gt_boxes_1
        blobs['im_info_1'] = np.array(
            [[im_blob_1.shape[2], im_blob_1.shape[3], im_scales_1[0]]],
            dtype=np.float32)
        gt_inds_2 = np.where(roidb[0]['gt_classes_2'] != 0)[0]
        gt_boxes_2 = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes_2[:, 0:4] = roidb[0]['boxes_2'][gt_inds_2, :] * im_scales_2[0]
        gt_boxes_2[:, 4] = roidb[0]['gt_classes_2'][gt_inds_2]
        blobs['gt_boxes_2'] = gt_boxes_2
        blobs['im_info_2'] = np.array(
            [[im_blob_2.shape[2], im_blob_2.shape[3], im_scales_2[0]]],
            dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob_1 = np.zeros((0, 5), dtype=np.float32)
        labels_blob_1 = np.zeros((0), dtype=np.float32)
        bbox_targets_blob_1 = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob_1 = np.zeros(bbox_targets_blob_1.shape, dtype=np.float32)
        rois_blob_2 = np.zeros((0, 5), dtype=np.float32)
        labels_blob_2 = np.zeros((0), dtype=np.float32)
        bbox_targets_blob_2 = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob_2 = np.zeros(bbox_targets_blob_2.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels_1, overlaps_1, im_rois_1, bbox_targets_1, bbox_inside_weights_1 \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                        num_clanum_classes, True)
            
            labels_2, overlaps_2, im_rois_2, bbox_targets_2, bbox_inside_weights_2 \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes, False)
            
            # Add to RoIs blob
            rois_1 = _project_im_rois(im_rois_1, im_scales_1[im_i])
            batch_ind_1 = im_i * np.ones((rois_1.shape[0], 1))
            rois_blob_this_image_1 = np.hstack((batch_ind_1, rois_1))
            rois_blob_1 = np.vstack((rois_blob_1, rois_blob_this_image_1))
            
            rois_2 = _project_im_rois(im_rois_2, im_scales_2[im_i])
            batch_ind_2 = im_i * np.ones((rois_2.shape[0], 1))
            rois_blob_this_image_2 = np.hstack((batch_ind_2, rois_2))
            rois_blob_2 = np.vstack((rois_blob_2, rois_blob_this_image_2))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob_1 = np.hstack((labels_blob_1, labels_1))
            bbox_targets_blob_1 = np.vstack((bbox_targets_blob_1, bbox_targets_1))
            bbox_inside_blob_1 = np.vstack((bbox_inside_blob_1, bbox_inside_weights_1))
            labels_blob_1 = np.hstack((labels_blob_1, labels_1))
            bbox_targets_blob_1 = np.vstack((bbox_targets_blob_1, bbox_targets_1))
            bbox_inside_blob_1 = np.vstack((bbox_inside_blob_1, bbox_inside_weights_1))

            labels_blob_2 = np.hstack((labels_blob_2, labels_2))
            bbox_targets_blob_2 = np.vstack((bbox_targets_blob_2, bbox_targets_2))
            bbox_inside_blob_2 = np.vstack((bbox_inside_blob_2, bbox_inside_weights_2))
            labels_blob_2 = np.hstack((labels_blob_2, labels_2))
            bbox_targets_blob_2 = np.vstack((bbox_targets_blob_2, bbox_targets_2))
            bbox_inside_blob_2 = np.vstack((bbox_inside_blob_2, bbox_inside_weights_2))
        
        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois_1'] = rois_blob_1
        blobs['labels_1'] = labels_blob_1
        blobs['rois_2'] = rois_blob_2
        blobs['labels_2'] = labels_blob_2

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets_1'] = bbox_targets_blob_1
            blobs['bbox_inside_weights_1'] = bbox_inside_blob_1
            blobs['bbox_outside_weights_1'] = \
                np.array(bbox_inside_blob_1 > 0).astype(np.float32)

            blobs['bbox_targets_2'] = bbox_targets_blob_2
            blobs['bbox_inside_weights_2'] = bbox_inside_blob_2
            blobs['bbox_outside_weights_2'] = \
                np.array(bbox_inside_blob_2 > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes, flag = True):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    if flag:
        labels = roidb['max_classes_1']
        overlaps = roidb['max_overlaps_1']
        rois = roidb['boxes_1']
    else:
        labels = roidb['max_classes_2']
        overlaps = roidb['max_overlaps_2']
        rois = roidb['boxes_2']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    if flag:
        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets_1'][keep_inds, :], num_classes)
    else: 
        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets_2'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im_1 = cv2.imread(roidb[i]['image_1'])
        if roidb[i]['flipped_1']:
            im_1 = im_1[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im_1, im_scale_1 = prep_im_for_blob(im_1, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)

        im_2 = cv2.imread(roidb[i]['image_2'])
        if roidb[i]['flipped_2']:
            im_2 = im_2[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im_2, im_scale_2 = prep_im_for_blob(im_2, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales_1.append(im_scale_1)
        processed_ims_1.append(im_1)
        im_scales_2.append(im_scale_2)
        processed_ims_2.append(im_2)

    # Create a blob to hold the input images
    blob_1 = im_list_to_blob(processed_ims_1)
    blob_2 = im_list_to_blob(processed_ims_2)

    return blob_1, blob_2, im_scales_1, im_scales_2

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
