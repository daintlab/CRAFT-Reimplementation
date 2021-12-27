"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

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
import wandb

from craft import CRAFT
from data.load_icdar import load_icdar2015_gt, load_icdar2013_gt, load_synthtext_gt
from utils.inference_boxes import test_net
from data import imgproc
from collections import OrderedDict
from metrics.eval_det_iou import DetectionIoUEvaluator

# config_defaults = {
#     'text_threshold': 0.7,
#     'low_text': 0.55,
#     'link_threshold': 0.23
# }
# wandb.init(project='ocr_craft')
# config = wandb.config


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict



def main(model, args, evaluator, data_li=''):
    model.eval()

    if data_li != '':
        total_imgs_bboxes_gt, total_img_path = load_synthtext_gt(args.synthData_dir, data_li=data_li)
    else:
        test_folder = args.test_folder

        if test_folder.split('/')[-1].lower() == 'icdar2013':
            total_imgs_bboxes_gt, total_img_path = load_icdar2013_gt(dataFolder=test_folder,
                                                                     isTraing=args.isTraingDataset)
        else:
            total_imgs_bboxes_gt, total_img_path = load_icdar2015_gt(dataFolder=test_folder,
                                                                     isTraing=args.isTraingDataset)

    total_img_bboxes_pre = []
    for k, img_path in enumerate(tqdm(total_img_path)):
        image = imgproc.loadImage(img_path)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(model,
                                             image,
                                             args.text_threshold,
                                             args.link_threshold,
                                             args.low_text,
                                             args.cuda,
                                             args.poly,
                                             args.canvas_size,
                                             args.mag_ratio)

        # # ---------------------------------------------------------------------------------------------------------------#

        #rnd_list = [1, 264, 135, 352, 481, 250, 355, 436, 45, 181, 98, 173, 267, 200, 79, 395]

        viz = True
        #if k in rnd_list:
           # viz = True

        if viz == True:

            height, width, channel = image.shape
            overlay_region = cv2.resize(score_text[0], (width, height))
            overlay_aff = cv2.resize(score_text[1], (width, height))

            overlay_region = cv2.addWeighted(image.copy(), 0.4, overlay_region, 0.6, 5)
            overlay_aff = cv2.addWeighted(image.copy(), 0.4, overlay_aff, 0.6, 5)
            outpath = os.path.join(os.path.join(args.results_dir, "test_output"), str(utils.config.ITER))
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # save overlay
            filename, file_ext = os.path.splitext(os.path.basename(img_path))
            overlay_region_file = outpath + "/res_" + filename + '_region.jpg'
            cv2.imwrite(overlay_region_file, overlay_region)

            filename, file_ext = os.path.splitext(os.path.basename(img_path))
            overlay_aff_file = outpath + "/res_" + filename + '_affi.jpg'
            cv2.imwrite(overlay_aff_file, overlay_aff)

            ori_image_path = outpath + "/res_" + filename + '.jpg'
            cv2.imwrite(ori_image_path,image)

        # # ---------------------------------------------------------------------------------------------------------------#

        for box in bboxes:
            box_info = {"points": None, "text": None, "ignore": None}
            box_info["points"] = box
            box_info["text"] = "###"
            box_info["ignore"] = False
            single_img_bbox.append(box_info)
        total_img_bboxes_pre.append(single_img_bbox)
    results = []
    for gt, pred in zip(total_imgs_bboxes_gt, total_img_bboxes_pre):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
    wandb.log({"precision": metrics['precision'], "recall": metrics['recall'], "hmean": metrics['hmean']})

    return metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model',
                        default='/nas/home/gmuffiness/model/ocr/daintlab-CRAFT-Reimplementation_v3/checkpoint_84000.pth',
                        type=str, help='pretrained model')
    # parser.add_argument('--trained_model',
    #                     default='/nas/home/jihyokim/jm/CRAFT-new-backtime92/exp/1216_exp/backnew-base-syn-1/CRAFT_clr_95000.pth',
    #                     type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=960, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--isTraingDataset', default=False, type=str2bool, help='test for traing or test data')
    parser.add_argument('--test_folder', default='/nas/datahub/ICDAR2013', type=str,
                        help='folder path to input images')
    parser.add_argument('--results_dir', default='/nas/home/gmuffiness/result/ocr/daintlab-CRAFT-Reimplementation/v3_icdar2013', type=str,
                        help='folder path to input images')

    args = parser.parse_args()
    # wandb.config.update(args)
    # load net
    net = CRAFT()     # initialize
    # wandb.watch(net)
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    net_param = torch.load(args.trained_model)


    try:
        net.load_state_dict(copyStateDict(net_param['craft']))
    except:
        net.load_state_dict(copyStateDict(net_param))


    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()
    evaluator = DetectionIoUEvaluator()

    main(net, args, evaluator)
