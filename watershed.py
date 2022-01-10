"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import time

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
from data import imgproc
import Polygon as plg

from craft import CRAFT
from collections import OrderedDict


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

def crop_image_by_bbox(image, box):
    w = (int)(np.linalg.norm(box[0] - box[1]))
    h = (int)(np.linalg.norm(box[0] - box[3]))
    width = w
    height = h
    if h > w * 1.5:
        width = h
        height = w
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
    else:
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

    warped = cv2.warpPerspective(image, M, (width, height))
    return warped



def watershed(image,region_score, viz):
    # new backtime code

    visual = viz
    # region_score = np.uint8(np.clip(region_score, 0, 1) * 255)

    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_score
    if visual:
        cv2.imwrite('exp/{}'.format('gray_w4.jpg'), gray)


    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)

    if visual:
        cv2.imwrite('exp/{}'.format('binary_w4.jpg'), binary)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    sure_bg = mb
    if visual:
        cv2.imwrite('exp/{}'.format('sure_bg_w4.jpg'), sure_bg)
    ret, sure_fg = cv2.threshold(gray, 0.6 * np.max(gray), 255, cv2.THRESH_BINARY)
    #sure_fg = cv2.dilate(sure_fg, kernel, iterations=1)
    if visual:
        cv2.imwrite('exp/{}'.format('sure_fg_w4.jpg'), sure_fg)

    surface_fg = np.uint8(sure_fg)
    surface_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(surface_bg, surface_fg)
    if visual:
        cv2.imwrite('exp/{}'.format('unknown_w4.jpg'), unknown)

    # 获取maskers,在markers中含有种子区域
    #ret, markers = cv2.connectedComponents(surface_fg)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)
    # 分水岭变换
    markers = labels + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [255, 0, 0]

    color_markers = np.uint8(markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    if visual:
        cv2.imwrite('exp/{}'.format('water_w4.jpg'), image)
        cv2.imwrite('exp/{}'.format('markers_w4.jpg'), color_markers)


    boxes = []
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
        # if visual:
        #     cv2.polylines(image, [np.array(box, dtype=np.int) * 2], True, (0, 255, 255), 1)
        #     cv2.imwrite('exp/{}'.format('water1.jpg'), image)

    #boxes = np.array(boxes) * 2
    #boxes = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))

    return np.array(boxes), color_markers



def watershed_v2(region_score, viz):

    # region_score = np.uint8(np.clip(region_score, 0, 1) * 255)

    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_score

    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, init_markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    init_markers = init_markers + 1
    # Now, mark the region of unknown with zero
    init_markers[unknown == 255] = 0
    init_markers_copy = init_markers.copy()

    dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX) * 255
    dist_transform = np.uint8(dist_transform)
    dist_transform = cv2.cvtColor(dist_transform, cv2.COLOR_GRAY2RGB)
    ret, dist_transform_binary = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    final_markers = cv2.watershed(dist_transform_binary, init_markers)
    ori_region_score[final_markers == -1] = [255, 0, 0]

    color_markers = np.uint8(final_markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    if viz:
        sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
        sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
        unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

        init_markers_copy = np.uint8(init_markers_copy + 1)
        init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
        init_markers_copy = np.uint8(init_markers_copy)
        init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)

        vis_result = np.vstack(
            [sure_bg_copy, dist_transform, sure_fg_copy, unknown_copy, init_markers_copy, dist_transform_binary,
             color_markers, ori_region_score])
        cv2.imwrite('exp/{}'.format('watershed_result.png'), vis_result)

    # make boxes
    boxes = []
    for i in range(2, np.max(final_markers) + 1):
        np_contours = np.roll(np.array(np.where(final_markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
        # if visual:
        #     cv2.polylines(image, [np.array(box, dtype=np.int) * 2], True, (0, 255, 255), 1)
        #     cv2.imwrite('exp/{}'.format('water1.jpg'), image)

    #boxes = np.array(boxes) * 2
    #boxes = sorted(boxes, key=lambda item: (item[0][0], item[0][1]))

    return np.array(boxes), color_markers


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()



    # render results (optional)
    render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # boxes1 = np.array([[1, 576], [44, 575], [35, 886], [2, 885]], dtype=np.int)
    # ret_score_text = crop_image_by_bbox(ret_score_text, boxes1//2)
    # cv2.polylines(ret_score_text, [boxes1//2], True, (0, 0, 255), 1)

    print(ret_score_text.shape)
    return ret_score_text



if __name__ == '__main__':
    use_cuda = True
    load_model_dir = "/home/yanhai/OCR/OCRRepo/craft/githubcraft/CRAFT-Reimplementation/weights_52000.pth"
    load_model_dir2 = "/home/yanhai/OCR/OCRRepo/craft/githubcraft/CRAFT-Reimplementation/craft_ic15_20k.pth"
    test_img_path = "/media/yanhai/disk21/SynthTextData/SynthText/138/punting_3_64.jpg"

    #hyp paras
    text_threshold = 0.7
    link_threshold = 0.4
    low_text = 0.4
    mag_ratio = 1.5
    poly = False
    refine_net = None
    canvas_size = 1280


    # load net
    net = CRAFT()     # initialize
    net1 = CRAFT()
    print('Loading weights from checkpoint')
    net.load_state_dict(copyStateDict(torch.load(load_model_dir)))
    net1.load_state_dict(copyStateDict(torch.load(load_model_dir2)))

    if use_cuda:
        net = net.cuda()
        net1 = net1.cuda()
        cudnn.benchmark = False

    net.eval()
    # load data
    print("load Test image!")
    image = imgproc.loadImage(test_img_path)

    score_text = test_net(net, image, text_threshold, link_threshold, low_text, use_cuda, poly, canvas_size, mag_ratio, refine_net)
    score_text1 = test_net(net1, image, text_threshold, link_threshold, low_text, use_cuda, poly, canvas_size, mag_ratio, refine_net)

    # bboxes = watershed1(score_text, False)
    # for boxes in bboxes:
    #     cv2.polylines(score_text, [boxes.astype(np.int)], True, (0, 255, 255), 1)
    stack_score = np.hstack((score_text, score_text1))
    cv2.namedWindow("water", cv2.WINDOW_NORMAL)
    cv2.imshow("water", stack_score)
    cv2.waitKey(0)
