import random

import scipy.io as scio
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import re
import itertools
import Polygon as plg
from PIL import Image

import utils.config
from data import imgproc
from utils import config

from gaussianMap.gaussian import GaussianTransformer
from data.boxEnlarge import enlargebox
from data.imgaug import random_scale, random_scale_for_synth, random_crop_v0, random_crop, random_crop_v2, random_horizontal_flip, random_rotate
from watershed import watershed, watershed1,  watershed4, watershed_v2, watershed_v3, watershed_v4
from data.pointClockOrder import mep
from utils import craft_utils


def saveInput(imagename, image, region_scores, affinity_scores, confidence_mask):
    image = np.uint8(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes, polys = craft_utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.85, 0.2, 0.5, False)
    boxes = np.array(boxes, np.int32) * 2
    if len(boxes) > 0:
        np.clip(boxes[:, :, 0], 0, image.shape[1])
        np.clip(boxes[:, :, 1], 0, image.shape[0])
        for box in boxes:
            cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask / 255)

    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))
    confidence_mask_gray = cv2.resize(confidence_mask_gray, (width, height))

    overlay_region = cv2.addWeighted(image, 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image, 0.4, overlay_aff, 0.7, 6)

    gt_scores = np.concatenate([overlay_region, overlay_aff], axis=1)
    confidence_mask_gray = np.concatenate([np.zeros_like(confidence_mask_gray), confidence_mask_gray], axis=1)

    output = np.concatenate([gt_scores, confidence_mask_gray], axis=1)

    output = np.hstack([image, output])

    outpath = os.path.join(os.path.join(config.RESULT_DIR, '{}/input'.format(str(config.ITER // 100))),
                           "%s_input.jpg" % imagename)
    #print(outpath)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    cv2.imwrite(outpath, output)


def saveImage(imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
    output_image = np.uint8(image.copy())
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    if len(bboxes) > 0:
        affinity_bboxes = np.int32(affinity_bboxes)
        for i in range(affinity_bboxes.shape[0]):
            cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
        for i in range(len(bboxes)):
            _bboxes = np.int32(bboxes[i])
            for j in range(_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, overlay_aff, 0.6, 5)

    heat_map = np.concatenate([overlay_region, overlay_aff], axis=1)
    output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)

    outpath = os.path.join(os.path.join(config.RESULT_DIR, '{}/input'.format(str(config.ITER // 100))), imagename)
    #print(outpath)
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    cv2.imwrite(outpath, output)



class SynthTextDataLoader(data.Dataset):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, viz=False, mode=''):
        assert 'synthtext' in data_dir_list.keys()

        self.target_size = target_size
        self.data_dir_list = data_dir_list
        self.viz = viz
        self.charbox, self.image, self.imgtxt = self.load_synthtext(mode)
        self.gen = GaussianTransformer(200, 1.5)
        # self.gen.gen_circle_mask()


    def choice_train_test_split(self,X, test_size=0.1, shuffle=True, random_state=1004):

        test_num = int(X.shape[0] * test_size)
        train_num = X.shape[0] - test_num

        if shuffle:
            np.random.seed(random_state)
            train_idx = np.random.choice(X.shape[0], train_num, replace=False)
            # -- way 1: using np.setdiff1d()
            test_idx = np.setdiff1d(range(X.shape[0]), train_idx)

            X_train = X[train_idx]
            X_test = X[test_idx]

        else:
            X_train = X[:train_num]
            X_test = X[train_num:]

        return X_train, X_test


    def load_synthtext(self, mode):

        gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        wordbox = gt['wordBB'][0]
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]

        trn_wordbox, tst_wordbox = self.choice_train_test_split(wordbox)
        trn_charbox, tst_charbox = self.choice_train_test_split(charbox)
        trn_image, tst_image = self.choice_train_test_split(image)
        trn_imgtxt, tst_imgtxt = self.choice_train_test_split(imgtxt)


        if mode == 'train':
            return trn_charbox, trn_image, trn_imgtxt

        elif mode == 'test':
            return tst_wordbox, tst_image, tst_imgtxt
        else:
            return charbox, image, imgtxt

    # def load_synthtext(self):
    #
    #     gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
    #     charbox = gt['charBB'][0]
    #     image = gt['imnames'][0]
    #     imgtxt = gt['txt'][0]
    #     return charbox, image, imgtxt


    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale_for_synth(image, _charbox, self.target_size)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]

            # fix negative coordinates
            # bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0, image.shape[1])
            # bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0, image.shape[0])

            assert len(bboxes) == len(words[i])
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, img_path

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences, img_path = self.load_synthtext_image_gt(index)


        #check negative coordinates
        # for cb in character_bboxes :
        #     if (cb < 0).astype('float32').sum() > 0 :
        #         import ipdb;
        #         ipdb.set_trace()



        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()


        if len(character_bboxes) > 0:
            region_scores = self.gen.generate_region(image.shape, character_bboxes, img_path)
            affinities_scores, affinity_bboxes = self.gen.generate_affinity(image.shape, character_bboxes, words)


        # if np.random.rand() > 0.99:
        #     print(np.random.random())
        #     self.viz = True

        if self.viz:
            saveImage(self.image[index][0], image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinities_scores,
                           confidence_mask)


        random_transforms = [image, region_scores, affinities_scores, confidence_mask*255]

        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        # if config.AUG == True:

            # random_transforms = random_horizontal_flip(random_transforms)
            # random_transforms = random_rotate(random_transforms)



        image, region_image, affinity_image, confidence_mask = random_transforms

        #resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            saveInput(self.image[index][0], image, region_image, affinity_image, confidence_mask)
            self.viz = False


        image = Image.fromarray(image)


        if config.AUG == True:
            image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
            #image = transforms.ColorJitter(brightness=32.0 / 255, contrast=0.5, saturation=0.5, hue=0.25)(image)
        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = image.transpose(2, 0, 1)



        region_image = region_image.astype(np.float32) / 255
        affinity_image = affinity_image.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) /255

        return image, region_image, affinity_image, confidence_mask, confidences

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.pull_item(index)





class ICDAR2015(data.Dataset):
    def __init__(self, net, icdar2015_folder, target_size=768, viz=False, sigma=40):

        self.net = net
        self.net.eval()

        self.target_size = target_size
        self.gen = GaussianTransformer(200, 1.5, sigma=sigma)

        self.img_folder = os.path.join(icdar2015_folder, 'ch4_training_images')
        self.gt_folder = os.path.join(icdar2015_folder, 'ch4_training_localization_transcription_gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)
        self.rnd_list = [189, 41, 723, 251, 232, 115, 634, 951, 247, 25, 400, 704, 619, 305, 423, 20, 31]
        # self.rnd_list = list(range(1000))
        self.viz = viz

    def __getitem__(self, index):
        return self.pull_saved_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]


    def load_gt(self, gt_path):


        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
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
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
            words.append(word)

        return bboxes, words

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def crop_image_by_bbox(self, image, box):


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
        return warped, M


    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len


    def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=False, imagename=''):

       # print('inference_pursedo_bboxes model last parameters
       # :{}'.format(net.module.conv_cls[-1].weight.reshape(2, -1)))
        if net.training:
            net.eval()
        with torch.no_grad():
            word_image, MM = self.crop_image_by_bbox(image, word_bbox)

            real_word_without_space = word.replace('\s', '')
            real_char_nums = len(real_word_without_space)
            input = word_image.copy()
            # 왜 64로 scale 조절을 하는 걸까?? --> https://github.com/clovaai/CRAFT-pytorch/issues/18
            scale = 64.0 / input.shape[0]
            #input = cv2.resize(input, None, fx=scale, fy=scale)
            input = cv2.resize(input, None, fx=scale, fy=scale)
            input_copy = input.copy()


            img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                       variance=(0.229, 0.224, 0.225)))
            img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
            img_torch = img_torch.type(torch.FloatTensor).cuda()
            scores, _ = net(img_torch)
            region_scores = scores[0, :, :, 0].cpu().data.numpy()
            region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
            bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
            bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2RGB)

            pursedo_bboxes, color_markers = watershed_v4(bgr_region_scores.copy(), input.copy(), viz=False)

            if len(pursedo_bboxes) > 0:

                pursedo_bboxes[:, :, 0] = np.clip(pursedo_bboxes[:, :, 0], 0, bgr_region_scores.shape[1])
                pursedo_bboxes[:, :, 1] = np.clip(pursedo_bboxes[:, :, 1], 0, bgr_region_scores.shape[0])



            _tmp = []
            # except for the small box
            for i in range(pursedo_bboxes.shape[0]):
                if np.mean(pursedo_bboxes[i].ravel()) > 2: # ravel -> 1차원 변환
                    _tmp.append(pursedo_bboxes[i])
                else:
                    print("filter bboxes", pursedo_bboxes[i]) # 작은 box들

                # check small box 2
                # import ipdb;ipdb.set_trace()
                #
                # poly = plg.Polygon(pursedo_bboxes[i])
                # area = poly.area()
                # if area < 10:
                #     continue
                # _tmp.append(pursedo_bboxes[i])
                #
                #
                #
                #
                #
                # pursedo_bboxes_ = pursedo_bboxes[i].copy()
                # top_left = np.array([np.min(pursedo_bboxes[i][:, 0]), np.min(pursedo_bboxes[i][:, 1])]).astype(np.int32)
                # pursedo_bboxes_ -= top_left[None, :]
                #
                # width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
                #     pursedo_bboxes_[:, 1]).astype(np.int32)
                #
                # if width >0 or height >0:
                #     _tmp.append(pursedo_bboxes[i])
                # else:
                #     import ipdb;ipdb.set_trace()
                #     print("filter bboxes", pursedo_bboxes[i])  # 작은 box들



            pursedo_bboxes = np.array(_tmp, np.float32)
            if pursedo_bboxes.shape[0] > 1:
                index = np.argsort(pursedo_bboxes[:, 0, 0])
                pursedo_bboxes = pursedo_bboxes[index]


            confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

            bboxes = []
            if confidence <= 0.5:  # confidence 값들이 낮은 경우 등분하고, 이떄 confidence 0.5
                width = input.shape[1]
                height = input.shape[0]

                width_per_char = width / len(word)
                for j, char in enumerate(word):
                    if char == ' ':
                        continue
                    left = j * width_per_char
                    right = (j + 1) * width_per_char
                    bbox = np.array([[left, 0], [right, 0], [right, height],
                                     [left, height]])
                    bboxes.append(bbox)

                bboxes = np.array(bboxes, np.float32)
                confidence = 0.5

            else:
                bboxes = pursedo_bboxes


            if viz == True:

               # -----------------------------------------------------------------------------------------------#

                input_copy1 = input_copy.copy()
                _purs_bboxes = np.int32(pursedo_bboxes.copy())
                if len(_purs_bboxes) > 0:
                    _purs_bboxes[:, :, 0] = np.clip(_purs_bboxes[:, :, 0], 0, input.shape[1])
                    _purs_bboxes[:, :, 1] = np.clip(_purs_bboxes[:, :, 1], 0, input.shape[0])
                    for bbox_p in _purs_bboxes:
                        cv2.polylines(np.uint8(input_copy1), [np.reshape(bbox_p, (-1, 1, 2))], True, (255, 0, 0))

                input_copy2 = input_copy.copy()
                _tmp_bboxes = np.int32(bboxes.copy())
                _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
                _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
                for bbox in _tmp_bboxes:
                    cv2.polylines(np.uint8(input_copy2), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))

                region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
                region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))

                # viz_image2 = np.hstack([input_copy[:, :, ::-1], region_scores_color, color_markers,
                #                        input_copy1[:, :, ::-1], input_copy2[:, :, ::-1]])
                # cv2.imwrite('/nas/home/gmuffiness/result/temp_hstack.jpg', viz_image2)

                #gaussian
                target = self.gen.generate_region(region_scores_color.shape, [_tmp_bboxes])
                target_color = cv2.applyColorMap(target.astype('uint8'), cv2.COLORMAP_JET)

                overlay_img = cv2.addWeighted(input_copy[:, :, ::-1], 0.7, target_color, 0.3, 5)
                # ori img , region score, watershed, box img
                viz_image = np.hstack([input_copy[:, :, ::-1], region_scores_color,color_markers,
                                       input_copy1[:, :, ::-1],input_copy2[:, :, ::-1], target_color, overlay_img])


                save_path = os.path.join(config.RESULT_DIR, str(config.ITER//100))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                cv2.imwrite(os.path.join(save_path, '{}_{}'.format(imagename, 'hstack.jpg')), viz_image)
                # if config.ITER == 0:
                # cv2.imwrite(os.path.join(os.path.join(save_path, 'ori_img_v3'), '{}_{}'.format(imagename, 'img.jpg')), input_copy[:, :, ::-1])
                # cv2.imwrite(os.path.join(os.path.join(save_path, 'region_score_v3'), '{}_{}'.format(imagename, 'region_score.jpg')), bgr_region_scores)

                viz=False

                # -----------------------------------------------------------------------------------------------#

            bboxes /= scale

            try: # problem
                for k in range(len(bboxes)):
                    ones = np.ones((4, 1))
                    tmp = np.concatenate([bboxes[k], ones], axis=-1)
                    I = np.matrix(MM).I
                    ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                    bboxes[k] = ori[:, :2]

            except Exception as e:
                print(e)



            bb1 = bboxes.copy()


            if len(bboxes) > 0:
                bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0, image.shape[0])
                bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0, image.shape[1])


            # for cb in bboxes:
            #     # if (cb < 0).astype('float32').sum() > 0:
            #     #     import ipdb;
            #
            #     #check 1
            #     poly = plg.Polygon(cb)
            #     area = poly.area()
            #     if area < 10:
            #         import ipdb;ipdb.set_trace()
            #
            #     # check 2
            #     pursedo_bboxes_ = cb.copy()
            #     top_left = np.array([np.min(pursedo_bboxes[:, 0]), np.min(pursedo_bboxes[:, 1])]).astype(np.int32)
            #     pursedo_bboxes_ -= top_left[None, :]
            #
            #     width, height = np.max(pursedo_bboxes_[:, 0]).astype(np.int32), np.max(
            #         pursedo_bboxes_[:, 1]).astype(np.int32)
            #
            #     if width >0 or height >0:
            #         pass
            #     else:
            #         import ipdb;ipdb.set_trace()
            #         print("filter bboxes", pursedo_bboxes[i])  # 작은 box들



        if not net.training:
            net.train()

        return bboxes, region_scores, confidence

    def get_confidence_by_contour(self, image, region_scores, word_bbox, word, new_imagename, vis=False):

        word_image, _ = self.crop_image_by_bbox(image, word_bbox)
        word_region_score, MM = self.crop_image_by_bbox(region_scores, word_bbox)

        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_region_score.copy()
        # 왜 64로 scale 조절을 하는 걸까?? --> https://github.com/clovaai/CRAFT-pytorch/issues/18
        scale = 64.0 / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)

        ret, binary = cv2.threshold(input, 0.6 * 255, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        confidence = self.get_confidence(real_char_nums, len(contours))

        if confidence <= 0.5:  # confidence 값들이 낮은 경우, confidence 0.5
            confidence = 0.5

        if vis:
            word_image = cv2.resize(word_image, None, fx=scale, fy=scale)
            word_region_score = cv2.resize(word_region_score, None, fx=scale, fy=scale)
            word_region_score = cv2.applyColorMap(np.uint8(word_region_score), cv2.COLORMAP_JET)

            word_region_score = word_region_score.copy()
            binary = binary.copy()
            word_region_score = cv2.cvtColor(word_region_score, cv2.COLOR_BGR2RGB)
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # import ipdb; ipdb.set_trace()
            vis_result = np.hstack([word_image, word_region_score, binary])
            cv2.imwrite(f'/nas/home/gmuffiness/workspace/ocr_related/daintlab-CRAFT-Reimplementation/craft_jm/results_dir/exp_official_craft_supervision_v1.2/contour_sample/{new_imagename}_{confidence}.jpg', vis_result)
        return confidence


    def load_image_gt_and_confidence_mask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''


        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words = self.load_gt(gt_path)

        word_bboxes = np.float32(word_bboxes)

        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        confidences = []
        new_imagename = ''

        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):

                if words[i] == '###' or len(words[i].strip()) == 0:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                    continue
                assert words[i] != '###' and len(words[i].strip()) != 0

                pursedo_viz = False
                if int(imagename.split('.')[0].split('_')[1]) in self.rnd_list:
                    pursedo_viz = True
                    new_imagename = imagename.split('.')[0] + '_' + str(i)

                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=pursedo_viz,
                                                                                               imagename=new_imagename)

                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)

        return image, character_bboxes, new_words, confidence_mask, confidences


    def load_image_gt_and_saved_confidence_mask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''


        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words = self.load_gt(gt_path)

        word_bboxes = np.float32(word_bboxes)

        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_scale(image, word_bboxes, self.target_size)
        confidences = []

        # load confidence_mask
        query_idx = int(self.get_imagename(index).split('.')[0].split('_')[1])
        saved_cf_mask_path = os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_cf_mask_thresh_0.6.jpg')
        confidence_mask = cv2.imread(saved_cf_mask_path, cv2.IMREAD_GRAYSCALE)
        confidence_mask = cv2.resize(confidence_mask, (image.shape[1], image.shape[0])).astype(np.float32)

        #-------------------------------------------------------------------------------------#

        # To make confidence_mask : 처음에만 실행될, save 할 confidence mask를 만드는 과정

        # confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        #
        # confidences = []
        # new_imagename = ''
        #
        # if len(word_bboxes) > 0:
        #     for i in range(len(word_bboxes)):
        #
        #         if words[i] == '###' or len(words[i].strip()) == 0:
        #             cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
        #             continue
        #         assert words[i] != '###' and len(words[i].strip()) != 0
        #
        #
        #         pursedo_viz = False
        #         if int(imagename.split('.')[0].split('_')[1]) in self.rnd_list :
        #             pursedo_viz = True
        #             new_imagename = imagename.split('.')[0] +'_'+str(i)
        #
        #         query_idx = int(self.get_imagename(index).split('.')[0].split('_')[1])
        #         saved_region_scores_path = os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_region.jpg')
        #         region_scores = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        #         region_scores = cv2.resize(region_scores, (image.shape[1], image.shape[0])).astype(np.float32)
        #
        #         confidence = self.get_confidence_by_contour(image, region_scores, word_bboxes[i], words[i], new_imagename, pursedo_viz)
        #         confidences.append(confidence)
        #         cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))

        return image, word_bboxes, confidence_mask, confidences

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences  = self.load_image_gt_and_confidence_mask(index)

        #check minus coordinate

        for cb in character_bboxes :
            if (cb < 0).astype('float32').sum() > 0 :
                import ipdb;ipdb.set_trace()
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()


        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinities_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if len(character_bboxes) > 0:
            region_scores = self.gen.generate_region(image.shape, character_bboxes)
            affinities_scores, affinity_bboxes = self.gen.generate_affinity(image.shape, character_bboxes, words)

        if int(self.get_imagename(index).split('.')[0].split('_')[1]) in self.rnd_list and \
                self.get_imagename(index).split('_')[0] == 'img':
            self.viz = True

        if self.viz:
            saveImage(self.get_imagename(index), image.copy(), character_bboxes.copy(), affinity_bboxes.copy(),
                      region_scores.copy(),affinities_scores.copy(),confidence_mask.copy())


        random_transforms = [image, region_scores, affinities_scores, confidence_mask*255]
        # randomcrop = eastrandomcropdata((768,768))
        # region_image, affinity_image, character_bboxes = randomcrop(region_image, affinity_image, character_bboxes)


        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        # random_transforms = random_crop_v2(random_transforms, (self.target_size, self.target_size))

        if config.AUG == True:
            random_transforms = random_horizontal_flip(random_transforms)
            random_transforms = random_rotate(random_transforms)
        image, region_image, affinity_image, confidence_mask = random_transforms

        # resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            saveInput(self.get_imagename(index), image.copy(), region_image.copy(),
                      affinity_image.copy(), confidence_mask.copy())

        image = Image.fromarray(image)

        if config.AUG == True:
            image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
            # image = transforms.GaussianBlur(kernel_size=(10, 10), sigma=(0.1, 5))(image)
        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = image.transpose(2, 0, 1)


        region_image = region_image.astype(np.float32) / 255
        affinity_image = affinity_image.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) /255

        self.viz = False


        return image, region_image, affinity_image, confidence_mask, confidences


    def pull_saved_item(self, index):

        query_idx = int(self.get_imagename(index).split('.')[0].split('_')[1])
        # print(f'current index : {query_idx}')
        image, word_bboxes, confidence_mask, confidences  = self.load_image_gt_and_saved_confidence_mask(index)

        # 기존 code 중 아래 random_crop에서 쓰이게 될 character bboxes 형식을 맞춰주기 위해, word bboxes를 1개의 character씩 담긴 bboxes로 만들어 줌
        character_bboxes = []
        trunc_mask = np.zeros([image.shape[0], image.shape[1]])
        for i in range(len(word_bboxes)):
            cv2.fillPoly(trunc_mask, [np.int32(word_bboxes[i])], 1)
            if (word_bboxes[i] < 0).sum() > 0:
                # trunc_mask_temp1 = trunc_mask.copy()
                # cv2.fillPoly(trunc_mask_temp1, [np.int32(word_bboxes[i])], 1)
                # cv2.imwrite('/nas/home/gmuffiness/result/trunc_before.png', (trunc_mask_temp1 * 255).astype(np.uint8))
                word_bboxes[i] = np.where(word_bboxes[i] < 0, 0, word_bboxes[i])
                # trunc_mask_temp2 = trunc_mask.copy()
                # cv2.fillPoly(trunc_mask_temp2, [np.int32(word_bboxes[i])], 1)
                # cv2.imwrite('/nas/home/gmuffiness/result/trunc_after.png', (trunc_mask_temp2 * 255).astype(np.uint8))
                # import ipdb;ipdb.set_trace()
            character_bboxes.append(np.expand_dims(word_bboxes[i], 0))
        # save confidence mask

        # confidence_mask_copy = (confidence_mask * 255).astype(np.uint8)
        # confidence_mask_copy = cv2.applyColorMap(confidence_mask_copy, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_cf_mask_jet_thresh_0.6.jpg'), confidence_mask_copy)
        # cv2.imwrite(os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_cf_mask_thresh_0.6.jpg'), confidence_mask)

        #check minus coordinate

        for cb in character_bboxes :
            if (cb < 0).astype('float32').sum() > 0 :
                import ipdb;ipdb.set_trace()
                print(query_idx)
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()

        # use official CRAFT model's output as teacher (to make pseudo-label)

        saved_region_scores_path = os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_region.jpg')
        saved_affi_scores_path = os.path.join(config.OFFICIAL_SUPERVISION_DIR, f'res_img_{query_idx}_affi.jpg')
        region_scores = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        affinities_scores = cv2.imread(saved_affi_scores_path, cv2.IMREAD_GRAYSCALE)
        affinity_bboxes = []
        region_scores = cv2.resize(region_scores, (image.shape[1], image.shape[0])).astype(np.float32)
        affinities_scores = cv2.resize(affinities_scores, (image.shape[1], image.shape[0])).astype(np.float32)

        # truncate region, affinity out of GT box
        trunc_mask = trunc_mask.astype(np.float32)
        region_scores = region_scores * trunc_mask
        affinities_scores = affinities_scores * trunc_mask

        if int(self.get_imagename(index).split('.')[0].split('_')[1]) in self.rnd_list and \
                self.get_imagename(index).split('_')[0] == 'img':
            self.viz = True

        if self.viz:
            saveImage(self.get_imagename(index), image.copy(), character_bboxes.copy(), affinity_bboxes.copy(),
                      region_scores.copy(),affinities_scores.copy(),confidence_mask.copy())


        random_transforms = [image, region_scores, affinities_scores, confidence_mask*255]
        # randomcrop = eastrandomcropdata((768,768))
        # region_image, affinity_image, character_bboxes = randomcrop(region_image, affinity_image, character_bboxes)


        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        # random_transforms = random_crop_v2(random_transforms, (self.target_size, self.target_size))

        if config.AUG == True:
            random_transforms = random_horizontal_flip(random_transforms)
            random_transforms = random_rotate(random_transforms)
        image, region_image, affinity_image, confidence_mask = random_transforms

        # resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)





        if self.viz:
            saveInput(self.get_imagename(index), image.copy(), region_image.copy(),
                      affinity_image.copy(), confidence_mask.copy())

        image = Image.fromarray(image)

        if config.AUG == True:
            image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = image.transpose(2, 0, 1)


        region_image = region_image.astype(np.float32) / 255
        affinity_image = affinity_image.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) /255

        self.viz = False


        return image, region_image, affinity_image, confidence_mask, confidences






if __name__ == "__main__":
    #data_dir_list = {"synthtext":"/media/yanhai/disk21/SynthTextData/SynthText"}
    #craft_data = SynthTextDataLoader(768, data_dir_list)

    from craft import CRAFT

    net = CRAFT()
    icdar2015_folder = '/home/data/ocr/detection/ICDAR2015'
    craft_data = ICDAR2015(net, icdar2015_folder, target_size = 768,)


    for index in range(10000):
        image, character_bboxes, words, confidence_mask, confidences, img_path = \
            craft_data.load_synthtext_image_gt(index)
        # # 测试
        # image = cv2.imread("/media/yanhai/disk21/SynthTextData/SynthText/8/ballet_106_0.jpg")
        # character_bboxes = np.array([[[[423.16126397,  22.26958901],
        #                              [438.2997574,   22.46075572],
        #                              [435.54895424,  40.15739982],
        #                              [420.17946701,  39.82138755]],
        #                             [[439.60847343,  21.60559248],
        #                              [452.61288403,  21.76391911],
        #                              [449.95797159,  40.47241401],
        #                              [436.74150236,  40.18347166]],
        #                             [[450.66887979,  27.0241972 ],
        #                              [466.31976402,  27.25747678],
        #                              [464.5848793,   40.79219178],
        #                              [448.74896556,  40.44598236]],
        #                             [[466.31976402,  27.25747678],
        #                              [482.22585715,  27.49456029],
        #                              [480.68235876,  41.14411963],
        #                              [464.5848793,   40.79219178]],
        #                             [[479.76190495,  27.45783459],
        #                              [498.3934528,   27.73554156],
        #                              [497.04793842,  41.50190876],
        #                              [478.18853922,  41.08959901]],
        #                             [[504.59927448,  28.73896576],
        #                              [512.20555863,  28.85582217],
        #                              [511.1101386,   41.80934074],
        #                              [503.4152019,   41.64111176]]]])
        # character_bboxes = character_bboxes.astype(np.int)
        gaussian_map = np.zeros(image.shape, dtype=np.uint8)
        gen = GaussianTransformer(200, 1.5)
        gen.gen_circle_mask()

        region_image = gen.generate_region(image.shape, character_bboxes)
        affinity_image, affinities = gen.generate_affinity(image.shape, character_bboxes, words)

        random_transforms = [image, region_image, affinity_image, confidence_mask]
        # randomCrop = EastRandomCropData((768,768))
        # region_image, affinity_image, character_bboxes = randomCrop(region_image, affinity_image, character_bboxes)


        random_transforms = random_crop(random_transforms, (768, 768), character_bboxes)
        image, region_image, affinity_image, confidence_mask = random_transforms
        region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
        affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)

        region_image = cv2.addWeighted(region_image, 0.3, image, 1.0, 0)
        affinity_image = cv2.addWeighted(affinity_image, 0.3, image, 1.0, 0)

        # cv2.imshow("gaussion", gaussion)
        # cv2.waitKey(0)
        for boxes in character_bboxes:
            for box in boxes:
                # image = cv2.imread(img_path)
                enlarge = enlargebox(box.astype(np.int), image.shape[0], image.shape[1])
                # print("enlarge:", enlarge)
                # gaussion = gen.generate_region(image.shape, np.array([[box]]))
                # gaussion = cv2.applyColorMap(gaussion, cv2.COLORMAP_JET)
        #         cv2.polylines(image, [enlarge], True, (0, 0, 255), 1)
        #         cv2.polylines(image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(region_image, [enlarge], True, (0, 0, 255), 1)
        #         cv2.polylines(region_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        # for box in affinities:
        #     cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 0, 255), 1)
        stack_image = np.hstack((region_image, affinity_image))
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", stack_image)
        cv2.waitKey(0)






