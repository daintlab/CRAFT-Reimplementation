import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import cv2
import time
import argparse
import numpy as np
import utils.config

from collections import OrderedDict
from data.dataset import SynthTextDataLoader, ICDAR2015
from data.imgproc import denormalizeMeanVariance

from craft import CRAFT
from loss.mseloss import Maploss, Maploss_v2, Maploss_v2_3, Maploss_v3, Maploss_v3_1
from torch.autograd import Variable
from utils.util import save_parser, make_logger, AverageMeter
from eval import main
from metrics.eval_det_iou import DetectionIoUEvaluator

import wandb

parser = argparse.ArgumentParser(description='CRAFT new-backtime92')

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser.add_argument('--results_dir', default='/data/workspace/woans0104/CRAFT-re-backtime92/exp/weekly_back_2', type=str,
                    help='Path to save checkpoints')
parser.add_argument('--synthData_dir', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument('--icdar2015_dir', default='/home/data/ocr/detection/ICDAR2015', type=str,
                    help='Path to root directory of icdar2015 dataset')
parser.add_argument("--ckpt_path", default='/nas/home/jihyokim/jm/CRAFT-new-backtime92/exp/1216_exp/backnew-base-syn-1/'
                                           'CRAFT_clr_100000.pth', type=str,
                    help="path to pretrained model")
parser.add_argument('--st_iter', default=0, type = int,
                    help='start iter')
parser.add_argument('--end_iter', default=25000, type = int,
                    help='end iter')

parser.add_argument('--icdar_batch', default=15, type = int,
                    help='batch size of icdar training')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_decay', default=10000, type=int, help='learning rate decay')
parser.add_argument('--gamma', '--gamma', default=0.8, type=float,
                    help='initial gamma')
parser.add_argument('--beta1', default=0.99, type=float, help='beta1(applied to momentum of Adam)')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2(applied to velocity of Adam)')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Weight decay')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--aug', default=False, type=str2bool, help='augmentation')
parser.add_argument('--amp', default=False, type=str2bool, help='Automatic Mixed Precision')
parser.add_argument('--loss_vis', default=True, type=str2bool, help='Option to visualize Loss calculate process')
parser.add_argument('--neg_rto', default=1, type=int, help='negative pixel ratio')
parser.add_argument('--enlargebox_mg', default=0.5, type=float, help='enlargebox_magine')

#for test

parser.add_argument('--trained_model', default='', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.85, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.5, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--isTraingDataset', default=False, type=str2bool, help='test for traing or test data')
parser.add_argument('--test_folder', default='/home/data/ocr/detection/ICDAR2015', type=str,
                    help='folder path to input images')

args = parser.parse_args()

wandb.init(project='ocr_craft_official_supervision')
wandb.run.name = args.results_dir[-4:] + '_train'
wandb.config.update(args)


def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return param_group['lr']


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

if __name__ == "__main__":

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    utils.config.RESULT_DIR = args.results_dir
    utils.config.AUG = args.aug
    utils.config.ENLARGEBOX_MAGINE = args.enlargebox_mg

    # 1. data load
    # 1-1. synthData load
    synthData_dir = {"synthtext": args.synthData_dir}
    synthDataLoader = SynthTextDataLoader(target_size=768, data_dir_list=synthData_dir, mode='total')

    train_syn_loder = torch.utils.data.DataLoader(
        synthDataLoader,
        batch_size=args.icdar_batch//5,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)
    batch_syn = iter(train_syn_loder)

    # 1-2. Real Data load
    craft = CRAFT(amp=args.amp)
    wandb.watch(craft)
    net_param = torch.load(args.ckpt_path)

    try:
        craft.load_state_dict(copyStateDict(net_param['craft']))
    except:
        craft.load_state_dict(copyStateDict(net_param))

    craft = torch.nn.DataParallel(craft).cuda()




    #print('init model last parameters :{}'.format(craft.module.conv_cls[-1].weight.reshape(2, -1)))

    realdata = ICDAR2015(craft, args.icdar2015_dir, target_size=768, viz=False)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=args.icdar_batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)


    # 2. optim & loss
    optimizer = optim.Adam(craft.parameters(), lr=args.lr,  betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if args.st_iter != 0:
        print('success optim_load')
        optimizer.load_state_dict(copyStateDict(net_param['optimizer']))
        args.st_iter = net_param['optimizer']['state'][0]['step']
        args.lr = net_param['optimizer']['param_groups'][0]['lr']

    criterion = Maploss_v2_3()


    # mixed precision
    if args.amp :
        if args.st_iter != 0 :
            scaler = torch.cuda.amp.GradScaler()
            scaler.load_state_dict(copyStateDict(net_param["scaler"]))
        else:
            scaler = torch.cuda.amp.GradScaler()

    # 3. training
    # logger
    trn_logger, val_logger = make_logger(path=args.results_dir)


    # save parser
    save_parser(args)


    train_step = args.st_iter
    whole_training_step = args.end_iter
    update_lr_rate_step = 0
    training_lr = args.lr
    loss_value = 0
    batch_time = 0
    losses = AverageMeter()


    while train_step < whole_training_step:

        for batch_index, (real_images, real_region_label, real_affi_label, real_confidence_mask, real_confidences) \
                in enumerate(real_data_loader):

            start_time = time.time()
            craft.train()
            if train_step>0 and train_step % args.lr_decay==0:
                update_lr_rate_step += 1
                training_lr = adjust_learning_rate(optimizer, args.gamma, update_lr_rate_step, args.lr)

            # syn image load
            syn_images, syn_region_label, syn_affi_label, syn_confidence_mask, _ = next(batch_syn)
            images = torch.cat((syn_images, real_images), 0)

            # ====================================== Save train image batches ======================================
            if args.loss_vis:
                imgs_vis = denormalizeMeanVariance(torch.einsum('nchw->nhwc', images).numpy()).reshape(-1,768,3)
                imgs_vis = cv2.cvtColor(imgs_vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'/nas/home/gmuffiness/result/vis_result_imgs_{batch_index}.png',imgs_vis)

                vis_result = cv2.cvtColor(imgs_vis, cv2.COLOR_BGR2RGB)
                image_log = wandb.Image(vis_result, caption=f"img_{batch_index}")
                wandb.log({f'img_{batch_index}': image_log})
            # ===============================================================================================
            # cat syn & real image
            region_image_label = torch.cat((syn_region_label, real_region_label), 0)
            affinity_image_label = torch.cat((syn_affi_label, real_affi_label), 0)
            mask = torch.cat((syn_confidence_mask, real_confidence_mask), 0)


            images = Variable(images).cuda()
            region_image_label = region_image_label.type(torch.FloatTensor)
            affinity_image_label = affinity_image_label.type(torch.FloatTensor)
            region_image_label = Variable(region_image_label).cuda()
            affinity_image_label = Variable(affinity_image_label).cuda()
            confidence_mask_label = Variable(mask).cuda()

            if args.amp:
                with torch.cuda.amp.autocast():
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    # loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)
                    loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label, args.neg_rto, vis=args.loss_vis, batch_index=batch_index)
            else:
                output, _ = craft(images)
                out1 = output[:, :, :, 0]
                out2 = output[:, :, :, 1]
                loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)




            optimizer.zero_grad()


            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()



            end_time = time.time()
            loss_value += loss.item()
            batch_time += (end_time - start_time)
            losses.update(loss.item(), images.size(0))

            if train_step > 0 and train_step%5==0:
                mean_loss = loss_value / 5
                loss_value = 0
                display_batch_time = time.time()
                avg_batch_time = batch_time/5
                batch_time = 0

                print("{}, training_step: {}|{}, learning rate: {:.8f}, training_loss: {:.5f}, avg_batch_time: {:.5f}"
                      .format(time.strftime('%Y-%m-%d:%H:%M:%S',time.localtime(time.time())), train_step,
                              whole_training_step, training_lr, mean_loss, avg_batch_time))
                wandb.log({'train_step': train_step, 'mean_loss': mean_loss})


            if train_step % 500 == 0 and train_step != 0:

                print('Saving state, index:', train_step)
                if args.amp:
                    torch.save({
                        'iter': train_step,
                        'craft': craft.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "scaler": scaler.state_dict()
                    }, args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth')


                else:
                    torch.save({
                        'iter': train_step,
                        'craft': craft.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth')


                evaluator = DetectionIoUEvaluator()

                try:
                    if args.amp:
                        metrics = main(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args,
                                       evaluator)
                    else:
                        metrics = main(args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth', args, evaluator)


                    val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])
                except:
                    val_logger.write([train_step, losses.avg, str(0)])

                losses.reset()


            train_step += 1
            utils.config.ITER = train_step
            if train_step >= whole_training_step: break


    #last model save
    if args.amp:
        torch.save({
            'iter': train_step,
            'craft': craft.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }, args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth')


    else:
        torch.save({
            'iter': train_step,
            'craft': craft.state_dict(),
            'optimizer': optimizer.state_dict()
        }, args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth')

    evaluator = DetectionIoUEvaluator()
    # metrics = main(craft, args, evaluator)

    if args.amp:
        metrics = main(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args,
                       evaluator)
    else:
        metrics = main(args.results_dir + '/CRAFT_clr_' + repr(train_step) + '.pth', args, evaluator)

    val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])