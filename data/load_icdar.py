
import re
import itertools
import cv2
import os
import numpy as np
from data.pointClockOrder import mep



def load_synthtext_gt(data_folder, data_li):


    dataFolder = data_folder


    wordbox, image, imgtxt = data_li

    total_img_path = []
    total_imgs_bboxes = []

    for index in range(len(wordbox)):
        img_path = os.path.join(dataFolder, image[index][0])
        total_img_path.append(img_path)
        try:
            _wordbox = wordbox[index].transpose((2, 1, 0))
        except:
            _wordbox = np.expand_dims(wordbox[index], axis=0)
            _wordbox = _wordbox.transpose((0, 2, 1))

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        if len(words) != len(_wordbox):
            import ipdb;ipdb.set_trace()

        single_img_bboxes = []
        for j in range(len(words)):
            boxInfos = {"points": None, "text": None, "ignore": None}
            boxInfos["points"] = _wordbox[j]
            boxInfos["text"] = words[j]
            boxInfos["ignore"] = False
            single_img_bboxes.append(boxInfos)

        total_imgs_bboxes.append(single_img_bboxes)


    return total_imgs_bboxes, total_img_path


def load_icdar2015_gt(dataFolder, isTraing=False):
    if isTraing:
        img_folderName = "icdar_c4_train_imgs"
        gt_folderName = "ch4_training_localization_transcription_gt"
    else:
        img_folderName = "ch4_test_images"
        gt_folderName = "ch4_test_localization_transcription_gt"


    gt_folder_path = os.listdir(os.path.join(dataFolder, gt_folderName))
    total_imgs_bboxes = []
    total_img_path = []
    for gt_path in gt_folder_path:
        gt_path = os.path.join(os.path.join(dataFolder, gt_folderName), gt_path)
        img_path = gt_path.replace(gt_folderName, img_folderName).replace(
            ".txt", ".jpg").replace("gt_", "")
        image = cv2.imread(img_path)
        lines = open(gt_path, encoding='utf-8').readlines()
        single_img_bboxes = []
        for line in lines:
            boxInfos = {"points": None, "text": None, "ignore": None}

            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            area, p0, p3, p2, p1, _, _ = mep(box)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            cv2.polylines(image, [np.array(new_box).astype(np.int)], True, (0, 0, 255), 1)
            boxInfos["points"] = new_box
            boxInfos["text"] = word
            if word == "###":
                boxInfos["ignore"] = True
            else:
                boxInfos["ignore"] = False

            single_img_bboxes.append(boxInfos)
        total_imgs_bboxes.append(single_img_bboxes)
        total_img_path.append(img_path)
    return total_imgs_bboxes, total_img_path, os.path.join(dataFolder, gt_folderName)

def load_icdar2013_gt(dataFolder, isTraing=False):
    if isTraing:
        img_folderName = "Challenge2_Test_Task12_Images"
        gt_folderName = "Challenge2_Test_Task1_GT"
    else:
        img_folderName = "Challenge2_Test_Task12_Images"
        gt_folderName = "Challenge2_Test_Task1_GT"


    gt_folder_path = os.listdir(os.path.join(dataFolder, gt_folderName))
    total_imgs_bboxes = []
    total_img_path = []
    for gt_path in gt_folder_path:
        gt_path = os.path.join(os.path.join(dataFolder, gt_folderName), gt_path)
        img_path = gt_path.replace(gt_folderName, img_folderName).replace(
            ".txt", ".jpg").replace("gt_", "")
        image = cv2.imread(img_path)
        lines = open(gt_path, encoding='utf-8').readlines()
        single_img_bboxes = []
        for line in lines:
            boxInfos = {"points": None, "text": None, "ignore": None}

            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(4)]
            word = ori_box[4:]
            word = ','.join(word)
            box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            # box = np.array(box, np.int32).reshape(4, 2)
            # area, p0, p3, p2, p1, _, _ = mep(box)
            #
            # bbox = np.array([p0, p1, p2, p3])
            # distance = 10000000
            # index = 0
            # for i in range(4):
            #     d = np.linalg.norm(box[0] - bbox[i])
            #     if distance > d:
            #         index = i
            #         distance = d
            # new_box = []
            # for i in range(index, index + 4):
            #     new_box.append(bbox[i % 4])
            # cv2.polylines(image, [np.array(new_box).astype(np.int)], True, (0, 0, 255), 1)
            boxInfos["points"] = box
            boxInfos["text"] = word
            if word == "###":
                boxInfos["ignore"] = True
            else:
                boxInfos["ignore"] = False

            single_img_bboxes.append(boxInfos)
        total_imgs_bboxes.append(single_img_bboxes)
        total_img_path.append(img_path)
    return total_imgs_bboxes, total_img_path, os.path.join(dataFolder, gt_folderName)