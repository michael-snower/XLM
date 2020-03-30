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

from .dictionary import Dictionary, SPECIAL_WORDS, SPECIAL_WORD

# from util.vis_helpers import make_vis_row, make_vis_col, load_image, cv2PIL
# from util.mask_helpers import split as split_img_and_mask
# from util.mask_helpers import get_enclosing_bbox

from pdb import set_trace as bp

logger = logging.getLogger(__name__)


# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]


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

def xyxy2xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


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

def angle_from_center(x, y, image_w, image_h):
    center_x = image_w // 2
    center_y = image_h // 2
    x_offset = center_x - x
    y_offset = center_y - y
    return angle_between(x_offset, y_offset)

def create_keypoint_dictionary(params):
    word2id = {}
    for pix_index in range(params.image_w * params.image_h):
        word2id['pix{}'.format(pix_index)] = pix_index
    # bos
    word2id['<s>'] = params.image_w * params.image_h
    # eos
    word2id['</s>'] = params.image_w * params.image_h + 1
    # pad
    word2id['<pad>'] = params.image_w * params.image_h + 2
    # unk
    word2id['<unk>'] = params.image_w * params.image_h + 3
    for i in range(4, SPECIAL_WORDS + 4):
        word2id[SPECIAL_WORD % params.image_w * params.image_h] = params.image_w * params.image_h + i
    id2word = {v: k for k, v in word2id.items()}
    counts = {k : i for i, k in enumerate(word2id.keys())}
    dico = Dictionary(id2word, word2id, counts)
    return dico

def get_keypoints(mask_path, num_keypoints, preprocessor, domain, same_size=True):
    mask = cv.imread(mask_path)
    if same_size is True:
        mask, bbox = preprocessor.process_mask(mask, domain)
    outline = cv.Laplacian(mask, cv.CV_32F)
    outline = outline.sum(axis=-1) / 3.
    contours, _ = cv.findContours(outline.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    keypoints = contours[0].squeeze()
    step = len(keypoints) // num_keypoints
    if step <= 0:
        return None, None, mask, bbox
    y = keypoints[::step, 0]
    x = keypoints[::step, 1]
    #angles = angle_from_center(x, y, preprocessor.image_w, preprocessor.image_h)
    # coord_ids = np.argsort(angles)
    # angles = np.sort(angles)
    # y = y[coord_ids]
    # x = x[coord_ids]
    keypoint_vis = np.zeros_like(outline, dtype=np.float32)
    for i, (_x, _y) in enumerate(zip(x, y)):
        color = i / 360. * 235. + 20.
        keypoint_vis[_x, _y] = color
    return (x, y), keypoint_vis, mask, bbox


# class KeypointDataset(BaseDataset):
#     def __init__(self, opt, mode):
#         self.dataroot = opt.dataroot
#         self.mode = mode
#         phase = opt.phase if mode == 'train' else opt.eval_phase
#         self.A_img_dir = os.path.join(opt.dataroot, phase + 'A')
#         self.A_mask_dir = os.path.join(opt.dataroot, phase + 'A_mask')
#         self.B_img_dir = os.path.join(opt.dataroot, phase + 'B')
#         self.B_mask_dir = os.path.join(opt.dataroot, phase + 'B_mask')
#         # image and masks paths
#         self.A_img_paths = make_dataset(self.A_img_dir) 
#         self.B_img_paths = make_dataset(self.B_img_dir)
#         self.A_mask_paths = [
#             os.path.abspath(os.path.join(self.A_mask_dir, mask_path))
#             mask_path for mask_path in os.listdir(self.A_mask_dir)
#         ]
#         self.B_mask_paths = [
#             os.path.abspath(os.path.join(self.B_mask_dir, mask_path))
#             mask_path for mask_path in os.listdir(self.B_mask_dir)
#         ]

#         def build_path_dicts(img_paths, mask_paths, domain):
#             img2mask_paths = defaultdict(list)
#             mask2img_paths = {}
#             for mask_path in mask_paths:
#                 mask_img_num = A_mask_name.split('_')[0]
#                 img_path = os.path.join(self.dataroot, phase + domain, mask_img_num + ".jpg")
#                 img2mask_paths[img_path].append(mask_path)
#                 mask2img_paths[mask_path] = img_path
#             return img2mask_paths, mask2img_paths

#         self.A_img2mask_paths, self.A_mask2img_paths = build_path_dicts(self.A_img_paths, self.A_mask_paths, "A")
#         self.B_img2mask_paths, self.B_mask2img_paths = build_path_dicts(self.B_img_paths, self.B_mask_paths, "B")

#         self.A_size = len(self.A_mask_paths)
#         self.B_size = len(self.B_mask_paths)

#         # preprocessing
#         self.canvas_w = opt.img_w
#         self.canvas_h = opt.img_h
#         self.UPP = UniformPreprocessor(self.canvas_w, self.canvas_h, self.dataroot)
#         self.data_augs = opt.data_augs
#         self.tensorize = T.ToTensor()
#         self.normalize = T.Normalize(
#             (0.5, 0.5, 0.5),
#             (0.5, 0.5, 0.5)
#         )

#         self.num_keypoints = 64

#     def name(self):
#         return 'UniformRGBMaskDataset'

#     def __len__(self):
#         return max(self.A_size, self.B_size)

#     # def vis_batch(self, data, i, vis_dir):

#     #     def process_domain(both, orig_img_path, orig_mask_path, labels):
#     #         img, mask = split_img_and_mask(both)
#     #         # process orig img and mask
#     #         orig_img = load_image(orig_img_path)
#     #         orig_mask = load_image(orig_mask_path)
#     #         domain_vis = make_vis_row(
#     #             [orig_img, img, orig_mask, mask],
#     #             labels=labels
#     #         )
#     #         return domain_vis

#     #     batch_size = len(data['A_both'])
#     #     A_batch_vis = []
#     #     B_batch_vis = []
#     #     for data_index in range(batch_size):
#     #         A_vis = process_domain(
#     #             data['A_both'][data_index : data_index + 1], 
#     #             data['A_img_path'][data_index], 
#     #             data['A_mask_path'][data_index],
#     #             labels=[
#     #                 'orig img A', 'transformed img A',
#     #                 'orig mask A', 'transformed mask A'
#     #             ]
#     #         )
#     #         B_vis = process_domain(
#     #             data['B_both'][data_index : data_index + 1],
#     #             data['B_img_path'][data_index],
#     #             data['B_mask_path'][data_index],
#     #             labels=[
#     #                 'orig img B', 'transformed img B',
#     #                 'orig mask B', 'transformed mask B'
#     #             ]
#     #         )
#     #         A_batch_vis.append(A_vis)
#     #         B_batch_vis.append(B_vis)
#     #     A_col = make_vis_col(A_batch_vis)
#     #     B_col = make_vis_col(B_batch_vis)
#     #     # save vis
#     #     vis_path = os.path.join(vis_dir, str(i) + '.png')
#     #     batch_vis = make_vis_row([A_col, B_col], hpadding=100)
#     #     logger.info('Saving vis to {}'.format(vis_path))
#     #     cv.imwrite(vis_path, batch_vis)

#     def _load_both_info(self, index, domain):
#         if domain == 'A':
#             mask_path = self.A_mask_paths[index]
#             img_path = self.A_mask2img_paths[mask_path]
#         else:
#             mask_path = self.B_mask_paths[index]
#             img_path = self.B_mask2img_paths[mask_path]
#         mask = Image.open(mask_path).convert('RGB')
#         img = Image.open(img_path).convert('RGB')
#         return mask, img, mask_path, img_path

#     def _aug(self, img, mask):
#         if 'h_flip' in self.data_augs and np.random.uniform() > 0.5:
#             img = Tf.hflip(img)
#             mask = Tf.hflip(mask)
#         return img, mask

#     def _get_keypoints(self, img, mask, domain, same_size=True):
#         if same_size is True:
#             img, mask, bbox = self.UPP.process(img, mask, domain)
#         outline = cv.Laplacian(mask, cv.CV_32F)
#         outline = outline.sum(axis=-1) / 3.
#         contours, _ = cv.findContours(outline.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#         outline_points = contours[0].squeeze()
#         step = len(outline_points) // self.num_keypoints
#         y = outline_points[::step, 0]
#         x = outline_points[::step, 1]
#         keypoints = np.zeros_like(outline, dtype=np.float32)
#         keypoints[x, y] = 255.
#         return outline, mask, img, bbox

#     def _get_train_item(self, index):
#         self.index = index
#         # choose random indices
#         A_index = index % self.A_size
#         B_index = random.randint(0, self.B_size - 1)
#         A_mask, A_img, A_mask_path, A_img_path = self._load_both_info(A_index, "A")
#         B_mask, B_img, B_mask_path, B_img_path = self._load_both_info(B_index, "B")
#         # extract keypoints
#         A_keypoints, A_mask_processed, A_img, A_bbox = self._get_keypoints(A_mask, A_img, "A")
#         B_keypoints, B_mask_processed, B_img, B_bbox = self._get_keypoints(B_mask, B_img, "B")
#         # img = self.tensorize(img)
#         # mask = self.tensorize(mask)
#         # img = self.normalize(img)
#         # mask = self.normalize(mask)
#         # mask = mask.sum(0, keepdim=True) / 3. # make mask 1 channel
#         # #both = torch.cat([img, mask], dim=0)
#         # if self.mode == "train":
#         #     img, mask = self._aug(img, mask)
#         return {}

#     def _get_eval_item(self, index):
#         A_index = index % self.A_size
#         B_index = index % self.B_size
#         A_img, A_boths, A_img_path, A_mask_paths, A_bboxes = self._extract_all_boths_from_img(A_index, domain='A')
#         B_img, B_boths, B_img_path, B_mask_paths, B_bboxes = self._extract_all_boths_from_img(B_index, domain='B')
#         return {
#             'A_boths'      : A_boths,
#             'A_mask_paths' : A_mask_paths,
#             'A_img_path'   : A_img_path,
#             'A_bboxes'     : A_bboxes,
#             'A_idx'        : A_index,
#             'B_boths'      : B_boths,
#             'B_mask_paths' : B_mask_paths,
#             'B_img_path'   : B_img_path,
#             'B_bboxes'     : B_bboxes,
#             'B_idx'        : B_index
#         }

#     def __getitem__(self, index):
#         self.index = index
#         if self.mode == 'train':
#             return self._get_train_item(index)
#         else:
#             return self._get_eval_item(index)
