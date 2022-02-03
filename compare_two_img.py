
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from tqdm import tqdm
import utils.config

from craft import CRAFT
from data.load_icdar import load_icdar2015_gt, load_icdar2013_gt, load_synthtext_gt
from utils.inference_boxes import test_net
from data import imgproc
from collections import OrderedDict
from metrics.eval_det_iou import DetectionIoUEvaluator

def compare_two_version(args):

    os.makedirs(os.path.join(args.results_dir, args.img_index), exist_ok=True)
    args.results_dir = os.path.join(args.results_dir, args.img_index)
    if len(args.v1_dir_name) != 75:
        v1_num = 'test'
        v2_num = args.v2_dir_name.split('_v')[1][:3]
    elif len(args.v2_dir_name) != 75:
        v1_num = args.v1_dir_name.split('_v')[1][:3]
        v2_num = 'test'
    else:
        v1_num = args.v1_dir_name.split('_v')[1][:3]
        v2_num = args.v2_dir_name.split('_v')[1][:3]

    print(f'compare 대상 v1 : {v1_num}, v2 : {v2_num}')
    dir1 = os.path.join(os.path.join(args.base_path, args.v1_dir_name), '0')
    dir2 = os.path.join(os.path.join(args.base_path, args.v2_dir_name), '0')

    img_name = f'res_img_{args.img_index}_box.jpg'
    img1_path = os.path.join(dir1, img_name)
    img2_path = os.path.join(dir2, img_name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    h, w, c = img1.shape

    save_file_name1 = 'v' + str(v1_num) + '_' + str(img_name)[:-8] + '_' + args.vis_option + '.jpg'
    save_file_name2 = 'v' + str(v2_num) + '_' + str(img_name)[:-8] + '_' + args.vis_option + '.jpg'
    print(f'vis option : {args.vis_option}')

    if args.vis_option == 'tt':
        cv2.imwrite(os.path.join(args.results_dir, save_file_name1), img1)
        cv2.imwrite(os.path.join(args.results_dir, save_file_name2), img2)
    elif args.vis_option == 'rg':
        cv2.imwrite(os.path.join(args.results_dir, save_file_name1), img1[h//2:,:w//2])
        cv2.imwrite(os.path.join(args.results_dir, save_file_name2), img2[h//2:,:w//2])
    elif args.vis_option == 'af':
        cv2.imwrite(os.path.join(args.results_dir, save_file_name1), img1[h//2:,w//2:])
        cv2.imwrite(os.path.join(args.results_dir, save_file_name2), img2[h//2:,w//2:])
    elif args.vis_option == 'box':
        cv2.imwrite(os.path.join(args.results_dir, save_file_name1), img1[:h//2,w//2:])
        cv2.imwrite(os.path.join(args.results_dir, save_file_name2), img2[:h//2,w//2:])
    elif args.vis_option == 'ori':
        cv2.imwrite(os.path.join(args.results_dir, save_file_name1), img1[:h//2,:w//2])
        cv2.imwrite(os.path.join(args.results_dir, save_file_name2), img2[:h//2,:w//2])

    print(f'{save_file_name1} file saved.')
    print(f'{save_file_name2} file saved.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--base_path', default='/nas/home/gmuffiness/result/ocr/icdar2015', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--v1_dir_name', default='official_test_output_aligned_official_hp_setting', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--v2_dir_name', default='exp_official_craft_supervision_v1.1_test_output_aligned_official_hp_setting', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--img_index', default='20', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--results_dir', default='/nas/home/gmuffiness/result/compare_two', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--vis_option', default='rg', type=str,
                        help='tt => Total set of img | rg => Region scores | af => Affinity scores | box => Pred box | ori => Ori img')

    args = parser.parse_args()
    compare_two_version(args)
