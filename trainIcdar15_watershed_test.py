import os
import random
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from gaussianMap.gaussian import GaussianTransformer


parser = argparse.ArgumentParser(description='CRAFT new-backtime92')
parser.add_argument('--results_dir', default='/data/workspace/woans0104/CRAFT-re-backtime92/exp/weekly_back_2', type=str,
                    help='Path to save checkpoints')
parser.add_argument('--data_dir', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of sample dataset')
args = parser.parse_args()




def watershed_v2(region_score, viz):
    region_scores_color = region_score.copy()
    region_scores_color = cv2.cvtColor(region_scores_color, cv2.COLOR_RGB2GRAY)
    region_scores_color = cv2.applyColorMap(np.uint8(region_scores_color), cv2.COLORMAP_JET)
    region_scores_color = cv2.cvtColor(region_scores_color, cv2.COLOR_BGR2RGB)
    ori_region_scores_color = region_scores_color.copy()

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
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, 0)

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
    #region_scores_color[final_markers == -1] = [255, 0, 0]

    color_markers = np.uint8(final_markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    # make boxes
    boxes = []
    for i in range(2, np.max(final_markers) + 1):
        x_min, x_max = np.min(np.where(final_markers == i)[1]), np.max(np.where(final_markers == i)[1])
        y_min, y_max = np.min(np.where(final_markers == i)[0]), np.max(np.where(final_markers == i)[0])
        # print(x_min, x_max, y_min, y_max)
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        box = np.array(box)
        boxes.append(box)

        #cv2.polylines(region_scores_color_viz, [np.array(box, dtype=np.int)], True, (0, 0, 255), 2)

    vis_result = [sure_bg.copy(), sure_fg.copy(), unknown.copy(), dist_transform_binary.copy(),color_markers.copy()]


    return np.array(boxes), vis_result




def viz(boxes, img, region, viz, img_path, mode = 'single'):

    bbox_input = img.copy()
    sure_bg, sure_fg, unknown, frame, watershed_maker = viz


    # for visualize on COLORMAP JET
    region_scores_color = region.copy()
    region_scores_color = cv2.applyColorMap(np.uint8(region_scores_color), cv2.COLORMAP_JET)


    sure_bg_copy = cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)
    sure_fg_copy = cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)
    unknown_copy = cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)

    if len(frame.shape) != 3:
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame_copy = frame

    init_markers_copy = np.uint8(watershed_maker + 1)
    init_markers_copy = init_markers_copy / (init_markers_copy.max() / 255)
    init_markers_copy = np.uint8(init_markers_copy)
    init_markers_copy = cv2.applyColorMap(init_markers_copy, cv2.COLORMAP_JET)
    init_markers_copy = cv2.cvtColor(init_markers_copy, cv2.COLOR_BGR2RGB)

    for bbox_p in boxes:
        cv2.polylines(np.uint8(bbox_input), [np.reshape(bbox_p, (-1, 1, 2))], True, (0, 0, 255))

    # gaussian
    gen = GaussianTransformer(200, 1.5, sigma=40)

    gauss_target = gen.generate_region(region_scores_color.shape, [boxes])
    gauss_target_color = cv2.applyColorMap(gauss_target.astype('uint8'), cv2.COLORMAP_JET)


    if mode == 'single':

        img_name, ext = os.path.splitext(img_path)


        cv2.imwrite('{}'.format(os.path.join(img_name+'_ori'+ext)), img)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_region' + ext)), region_scores_color)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_back' + ext)), sure_bg_copy)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_fore' + ext)), sure_fg_copy)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_unknown' + ext)), unknown_copy)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_frame' + ext)), frame_copy)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_watershed' + ext)), init_markers_copy)
        cv2.imwrite('{}'.format(os.path.join(img_name + '_bbox_result' + ext)), bbox_input)


    else :
        vis_result = np.hstack(
            [img, region_scores_color, sure_bg_copy, sure_fg_copy, unknown_copy, frame_copy, init_markers_copy,
             bbox_input,gauss_target_color])


        cv2.imwrite('{}'.format(img_path), vis_result)



if __name__ == "__main__":

    # make result dir
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    # load full img, full region score
    image_path = args.data_dir
    file_names = os.listdir(image_path)

    img_list = sorted([file for file in file_names if file.split('_')[-1] == "ori.jpg"])
    region_list = sorted([file for file in file_names if file.split('_')[-1] == "score.jpg"])


    for i in range(len(img_list)):

        assert len(img_list) == len(region_list)
        assert img_list[i][:9] == region_list[i][:9] #'img_115_1'=='img_115_1',

        # load img
        img = os.path.join(image_path, img_list[i])
        img_arr = cv2.imread(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        # load region score
        region_score = os.path.join(image_path, region_list[i])
        region_arr = cv2.imread(region_score, cv2.cv2.IMREAD_GRAYSCALE)

        # convert region score (gray -> rgb)
        bgr_region_scores = cv2.resize(region_arr, (img_arr.shape[1], img_arr.shape[0]))
        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2RGB)

        # watershed
        pursedo_bboxes, vis_result = watershed_v2(bgr_region_scores.copy(), True)

        # save img
        save_img_path = '{}_{}.jpg'.format(os.path.join(args.results_dir, img_list[i][:9]), 'watershed_jm')
        viz(pursedo_bboxes, img_arr, bgr_region_scores, vis_result, save_img_path, mode='single')




