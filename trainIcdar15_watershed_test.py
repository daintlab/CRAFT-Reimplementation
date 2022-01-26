import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import Polygon as plg
import matplotlib.pyplot as plt

from gaussianMap.gaussian import GaussianTransformer


parser = argparse.ArgumentParser(description='Watershed Visualisation')
parser.add_argument('--mode', 
                    default='hstack', 
                    type=str, 
                    help='\'single\' saves intermediate images individually, otherwise it saves all intermediate images to the one horizontally stacked image')
parser.add_argument('--results-dir', 
                    default='./exp/watershed-vis/', 
                    type=str, 
                    help='Path where the result images will be saved')
parser.add_argument('--img-dir', 
                    default='/nas/home/jihyokim/jm/CRAFT-new-backtime92/exp/0117_exp/watershed_sample/', 
                    type=str, 
                    help='Path where the test images are located')
parser.add_argument('--enlarge-ratio', 
                    default=-1.0, 
                    type=float, 
                    help='character bbox enlarge ratio, e.g., it would be set to 0.5 if you want to enlarge the box with ratio 1.5, in case of 1.75 it would be 0.75')
parser.add_argument('--whose',
                    default='jaemoon', 
                    type=str, 
                    help='the watershed algorithm which will be used to test')
args = parser.parse_args()


def pointAngle(Apoint, Bpoint):
    angle = (Bpoint[1] - Apoint[1]) / ((Bpoint[0] - Apoint[0]) + 10e-8)
    return angle

def pointDistance(Apoint, Bpoint):
    return math.sqrt((Bpoint[1] - Apoint[1])**2 + (Bpoint[0] - Apoint[0])**2)

def lineBiasAndK(Apoint, Bpoint):

    K = pointAngle(Apoint, Bpoint)
    B = Apoint[1] - K*Apoint[0]
    return K, B

def getX(K, B, Ypoint):
    return int((Ypoint-B)/K)

def sidePoint(Apoint, Bpoint, h, w, placehold, enlarge_ratio=0.5):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = enlarge_ratio * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

def watershed_dj_01(region_score):
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
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        box = np.array(box)
        boxes.append(box)

    vis_result = [sure_bg.copy(), sure_fg.copy(), unknown.copy(), dist_transform_binary.copy(), color_markers.copy()]

    return np.array(boxes), vis_result


def watershed_dj_02(region_score):

    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_BGR2GRAY)
    else:
        gray = region_score

    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
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
    ret, dist_transform_binary = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    final_markers = cv2.watershed(dist_transform_binary, init_markers)
    region_score[final_markers == -1] = [255, 0, 0]

    color_markers = np.uint8(final_markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    # make boxes
    boxes = []
    for i in range(2, np.max(final_markers) + 1):
        x_min, x_max = np.min(np.where(final_markers == i)[1]), np.max(np.where(final_markers == i)[1])
        y_min, y_max = np.min(np.where(final_markers == i)[0]), np.max(np.where(final_markers == i)[0])
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        box = np.array(box)
        boxes.append(box)

    vis_result = [sure_bg.copy(), sure_fg.copy(), unknown.copy(), dist_transform_binary.copy(), color_markers.copy()]

    return np.array(boxes), vis_result


def watershed_jm(region_score):
    # jm watershed

    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_RGB2GRAY)
    else:
        gray = region_score

    # 1. binary
    ret, frame = cv2.threshold(gray, 0.3 * np.max(gray), 255, cv2.THRESH_BINARY)
    ret, background = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    ret, foreground = cv2.threshold(gray, 0.6 * np.max(gray), 255, cv2.THRESH_BINARY)

    # 2. opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_fr = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)
    open_bg = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel, iterations=2)
    open_fg = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. surface background,foreground, unknown
    surface_fr = cv2.dilate(open_fr, kernel, iterations=1)
    surface_bg = cv2.dilate(open_bg, kernel, iterations=2)
    surface_fg = open_fg
    surface_fg = np.uint8(surface_fg)
    surface_bg = np.uint8(surface_bg)
    unknown = cv2.subtract(surface_bg, surface_fg)

    # 4. watershed
    ret, markers = cv2.connectedComponents(surface_fg)
    final_frame = cv2.cvtColor(surface_fr, cv2.COLOR_GRAY2RGB)

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(final_frame, markers=markers) # final_frame
    final_frame[markers == -1] = [0, 0, 255]

    color_markers = np.uint8(markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    boxes = []
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)

        x, y, w, h = cv2.boundingRect(np_contours)
        box =[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)

    vis_result = [surface_bg.copy(), surface_fg.copy(), unknown.copy(), final_frame.copy(), color_markers.copy()]

    return np.array(boxes), vis_result

# mine
def _watershed(region_score):

    ori_region_score = region_score.copy()

    if len(region_score.shape) == 3:
        gray = cv2.cvtColor(region_score, cv2.COLOR_RGB2GRAY)
    else:
        gray = region_score

    ''' 1. binary '''
    ret, bg_thresh = cv2.threshold(gray, 0.2*np.max(gray), 255, cv2.THRESH_BINARY)
    ret, fg_thresh = cv2.threshold(gray, 0.6*np.max(gray), 255, cv2.THRESH_BINARY)

    ''' 2. background noise removal '''
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bg_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure_bg = opening
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_bg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

    ''' 3. Find contours '''
    contours, hierachy = cv2.findContours(fg_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    ''' 4. Make blank image and draw diamond shapes on the blank image '''
    blank_image = np.zeros(region_score.shape, np.uint8)
    for one_contour in contours:
        one_contour = np.squeeze(one_contour, axis=1)
        idx = list(range(0, len(one_contour), int(len(one_contour) / 4)))
        coordinates = one_contour[idx]
        cv2.polylines(blank_image, [coordinates], True, (255, 255, 255), 1)
        cv2.fillPoly(blank_image, [coordinates], (255, 255, 255))
    blank_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    ret, sure_fg = cv2.threshold(blank_gray, 0, 255, cv2.THRESH_BINARY)

    ''' 5. Marker '''
    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    ''' 6. Finding unknown region '''
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ''' 7. watershed '''
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(region_score, markers)  # (img, markers)
    region_score[markers == -1] = [255, 0, 0] # [0, 0, 255]

    color_markers = np.uint8(markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    boxes = []
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)

        x, y, w, h = cv2.boundingRect(np_contours)
        box =[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)

    vis_result = [sure_bg.copy(), sure_fg.copy(), unknown.copy(), region_score.copy(), color_markers.copy()]

    return np.array(boxes), vis_result



def viz(boxes, img, region, viz, img_path, enlarge_ratio = -1.0, mode='hstack'):

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

    dict_bbox, dict_gaus = {}, {}
    
    for enlarge_ratio in [-1.0, 0.5, 0.75]:
        bbox_input_copy = bbox_input.copy()

        for i, bbox_p in enumerate(boxes):

            if enlarge_ratio != -1.0:
                # Enlarge
                Apoint, Bpoint, Cpoint, Dpoint = bbox_p
                K1, B1 = lineBiasAndK(bbox_p[0], bbox_p[2])
                K2, B2 = lineBiasAndK(bbox_p[3], bbox_p[1])
                X = (B2 - B1) / (K1 - K2)
                Y = K1 * X + B1
                center = [X, Y]

                h, w = img.shape[0], img.shape[1]
                x1, y1 = sidePoint(Apoint, center, h, w, 'leftTop', enlarge_ratio)
                x2, y2 = sidePoint(center, Bpoint, h, w, 'rightTop', enlarge_ratio)
                x3, y3 = sidePoint(center, Cpoint, h, w, 'rightBottom', enlarge_ratio)
                x4, y4 = sidePoint(Dpoint, center, h, w, 'leftBottom', enlarge_ratio)
                bbox_p = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                boxes[i] = bbox_p

            cv2.polylines(np.uint8(bbox_input_copy), [np.reshape(bbox_p, (-1, 1, 2))], True, (0, 0, 255))

        ''' 
        must check the enlarge function in GaussianTransformer
        
        gaussianMap.gaussian.py
        class GaussianTransformer():
          def add_character():
            bbox_copy = bbox.copy()
            bbox = enlargebox(bbox, image.shape[0], image.shape[1]) <- this line
        '''
        # gaussian
        gen = GaussianTransformer(200, 1.5, sigma=40)

        gauss_target = gen.generate_region(region_scores_color.shape, [boxes])
        gauss_target_color = cv2.applyColorMap(gauss_target.astype('uint8'), cv2.COLORMAP_JET)

        dict_bbox.update({enlarge_ratio+1.0: bbox_input_copy})
        dict_gaus.update({enlarge_ratio+1.0: gauss_target_color})

    if mode == 'single':

        img_person, ext = os.path.splitext(img_path.split('/')[-1])
        img_name, person = img_person.split('-')

        save_path = os.path.join(args.results_dir, 'single')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cv2.imwrite(os.path.join(save_path, f'{img_name}-original-{person}{ext}'), img)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-region-{person}{ext}'), region_scores_color)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-background-{person}{ext}'), sure_bg_copy)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-foreground-{person}{ext}'), sure_fg_copy)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-unknown-{person}{ext}'), unknown_copy)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-frame-{person}{ext}'), frame_copy)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-watershed-{person}{ext}'), init_markers_copy)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-bbox-enlarge-{enlarge_ratio+1}-{person}{ext}'), bbox_input)
        cv2.imwrite(os.path.join(save_path, f'{img_name}-gaus-mapping-{enlarge_ratio+1}-{person}{ext}'), gauss_target_color)

    else :
        vis_result = np.hstack(
            [img, region_scores_color, sure_bg_copy, sure_fg_copy, unknown_copy, frame_copy, init_markers_copy,
             dict_bbox[0.0], dict_gaus[0.0], dict_bbox[1.5], dict_gaus[1.5], dict_bbox[1.75], dict_gaus[1.75],])
            # [img, region_scores_color, sure_bg_copy, sure_fg_copy, unknown_copy, frame_copy, init_markers_copy,
            #  bbox_input, gauss_target_color]

        save_path = os.path.join(args.results_dir, 'hstack')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cv2.imwrite(f"{save_path}/{img_path.split('/')[-1]}", vis_result)



if __name__ == "__main__":

    # make result dir
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    if args.whose == 'jaemoon':
        watershed = watershed_jm
    elif args.whose == 'daejeong-01':
        watershed = watershed_dj_01
    elif args.whose == 'daejeong-02':
        watershed = watershed_dj_02
    elif args.whose == 'jihyo':
        watershed = _watershed

    # load full img, full region score
    image_path = args.img_dir
    folder_names = os.listdir(image_path)

    img_list = sorted([f'{folder}/{imgs}' for folder in folder_names for imgs in os.listdir(f'{image_path}/{folder}/') if imgs.split('_')[-1] == "ori.jpg"])
    region_list = sorted([f'{folder}/{imgs}' for folder in folder_names for imgs in os.listdir(f'{image_path}/{folder}/') if imgs.split('_')[-1] == "score.jpg"])

    for i in range(len(img_list)):

        assert len(img_list) == len(region_list)
        assert img_list[i][:9] == region_list[i][:9] #'img_115_1'=='img_115_1',

        # load img
        img = os.path.join(image_path, img_list[i])
        img_arr = cv2.imread(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        # load region score
        region_score = os.path.join(image_path, region_list[i])
        region_arr = cv2.imread(region_score, cv2.IMREAD_GRAYSCALE)

        # convert region score (gray -> rgb)
        bgr_region_scores = cv2.resize(region_arr, (img_arr.shape[1], img_arr.shape[0]))
        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2RGB)

        # watershed
        pursedo_bboxes, vis_result = watershed(bgr_region_scores.copy())

        # save img
        save_img_path = '{}-{}.jpg'.format(os.path.join(args.results_dir, img_list[i].split('/')[-1][:-8]), args.whose)
        viz(pursedo_bboxes, img_arr, bgr_region_scores, vis_result, save_img_path, enlarge_ratio=args.enlarge_ratio, mode=args.mode)
