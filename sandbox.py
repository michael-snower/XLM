import cv2 as cv
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as Tf
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev

import os
import os.path
from collections import defaultdict
from glob import glob
import random
import logging

from pdb import set_trace as bp

SAMPLE_DIR = "data_samples"
SAVE_DIR = "sandbox_output"
NUM_KEYPOINTS = 128
IMAGE_W = 512
IMAGE_H = 512

def get_enclosing_bbox(mask, thresh=0, bbox_format='xyxy', vis=False):
    mask = mask.sum(-1) / float(mask.shape[-1])
    mask_indices = np.argwhere(mask > thresh)
    if len(mask_indices) > 0:
        min_y, min_x = np.min(mask_indices, axis=0)
        max_y, max_x = np.max(mask_indices, axis=0)
    else:
        min_y, min_x = 0, 0
        max_y, max_x = 1, 1
    if vis:
        mask = mask.copy()
        mask = np.stack([mask] * 3, axis=-1)
        cv.rectangle(
            mask,
            (min_x, min_y),
            (max_x, max_y),
            (255, 0, 0), 
            2
        )
        cv.rectangle(
            mask,
            (0, 0),
            (199, 199),
            (0, 255, 0),
            1
        )
        cv.imwrite('test.png', mask)
    if bbox_format == 'xyxy':
        return [min_x, min_y, max_x, max_y]
    elif bbox_format == 'xywh':
        return xyxy2xywh(min_x, min_y, max_x, max_y)


class UniformPreprocessor(object):
    def __init__(self, canvas_w, canvas_h, dataset_name, same_size=False):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        if same_size is False:
            A_size_multiple, B_size_multiple = self.get_dataset_sizes(dataset_name)
        else:
            A_size_multiple = B_size_multiple = 0.9
        self.domain_A_size = self.canvas_w * A_size_multiple
        self.domain_B_size = self.canvas_w * B_size_multiple

    @staticmethod
    def get_dataset_sizes(dataset_name):
        if 'shp2gir' in dataset_name:
            A_size_multiple, B_size_multiple = 0.375, 0.75
        else:
            raise ValueError('Dataset name not recognized.')
        return A_size_multiple, B_size_multiple

    @staticmethod
    def resize_and_maintain_aspect_ratio(data, long_edge_target):
        img_h, img_w, *_ = np.asarray(data).shape
        if img_h > img_w:
            long_edge = img_h
            short_edge = img_w
            long_edge_is_h = True
        else:
            long_edge = img_w
            short_edge = img_h
            long_edge_is_h = False
        scale_factor = float(long_edge_target) / long_edge
        short_edge_target = short_edge * scale_factor
        if long_edge_is_h is True:
            target_dims = (int(short_edge_target), int(long_edge_target))
        else:
            target_dims = (int(long_edge_target), int(short_edge_target))
        data = cv.resize(data, target_dims, cv.INTER_CUBIC)
        return data

    def _split_diff(self, diff):
        if diff % 2 != 0:
            first_border = diff // 2
            second_border = first_border + 1
        else:
            first_border = second_border = diff // 2
        return first_border, second_border

    def _crop_and_pad(self, data, bbox, domain):
        #logger.info('bbox: {}'.format(str(bbox)))
        x1, y1, x2, y2 = bbox
        cropped = data[y1 : y2, x1 : x2, :]
        if domain == 'A':
            cropped = self.resize_and_maintain_aspect_ratio(cropped, self.domain_A_size)
        else:
            cropped = self.resize_and_maintain_aspect_ratio(cropped, self.domain_B_size)
        vertical_diff = self.canvas_h - cropped.shape[0]
        horizontal_diff = self.canvas_w - cropped.shape[1]
        # reduce size if necessary
        if vertical_diff < 0 or horizontal_diff < 0:
            logger.error('Instance too large.')
        top_border_size, bottom_border_size = self._split_diff(vertical_diff)
        left_border_size, right_border_size = self._split_diff(horizontal_diff)
        out = cv.copyMakeBorder(
            cropped,
            top=top_border_size,
            bottom=bottom_border_size,
            left=left_border_size,
            right=right_border_size,
            borderType=cv.BORDER_CONSTANT,
            value=[0., 0., 0.]
        )
        return out

    def process(self, img, mask, domain):
        #logger.info(index)
        # convert to opencv format
        img = np.array(img, np.float32)[..., ::-1]
        mask = np.array(mask, np.float32)[..., ::-1]
        #logger.info('img shape: {}, mask shape: {}'.format(img.shape, mask.shape))
        img[mask == 0] = 0
        bbox = get_enclosing_bbox(mask)
        # crop and pad so each is the proper size
        img = self._crop_and_pad(img, bbox, domain)
        mask = self._crop_and_pad(mask, bbox, domain)
        #cv.imwrite('img_{}.png'.format(index), img)
        #cv.imwrite('mask_{}.png'.format(mask), mask)
        # convert back to PIL
        img = cv2PIL(img)
        mask = cv2PIL(mask)
        return img, mask, bbox

    def process_mask(self, mask, domain):
        bbox = get_enclosing_bbox(mask)
        mask = self._crop_and_pad(mask, bbox, domain)
        return mask, bbox

def angle_between(x, y):
    ang = np.arctan2(y, x)
    return np.rad2deg((ang) % (2 * np.pi))

def coordinate_from_center(x, y, image_w, image_h):
    center_x = image_w // 2
    center_y = image_h // 2
    x_offset = center_x - x
    y_offset = center_y - y
    return angle_between(x_offset, y_offset)

def get_keypoints(mask, mask_name, num_keypoints):
    outline = cv.Laplacian(mask, cv.CV_32F)
    outline_vis_path = os.path.join(SAVE_DIR, mask_name + "_outline.png")
    cv.imwrite(outline_vis_path, outline)
    outline = outline.sum(axis=-1) / 3.
    contours, _ = cv.findContours(outline.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    outline_points = contours[0].squeeze()
    step = len(outline_points) // NUM_KEYPOINTS
    y = outline_points[::step, 0]
    x = outline_points[::step, 1]
    angles = coordinate_from_center(x, y, IMAGE_W, IMAGE_H)
    # coord_ids = np.argsort(angles)
    # angles = np.sort(angles)
    # y = y[coord_ids]
    # x = x[coord_ids]
    outline[...] = 0.
    for i, (_x, _y) in enumerate(zip(x, y)):
        color = i / 360. * 235. + 20.
        outline[_x, _y] = color
    keypoints_vis_path = os.path.join(SAVE_DIR, mask_name + "_keypoints.png")
    cv.imwrite(keypoints_vis_path, outline)

def main():
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    mask_filenames = os.listdir(SAMPLE_DIR)
    Processor = UniformPreprocessor(512, 512, "shp2gir")
    for mask_filename in mask_filenames:
        print("Processing {}".format(mask_filename))
        mask_path = os.path.abspath(os.path.join(SAMPLE_DIR, mask_filename))
        mask_name = mask_filename.split(".")[0]
        mask = cv.imread(mask_path)
        mask, _ = Processor.process_mask(mask, "B")
        processed_mask_vis_path = os.path.join(SAVE_DIR, mask_name + "_processed.png")
        cv.imwrite(processed_mask_vis_path, mask)
        get_keypoints(mask, mask_name, NUM_KEYPOINTS)


if __name__ == "__main__":
    main()