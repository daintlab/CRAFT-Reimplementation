import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import cv2
import time
import argparse
import numpy as np

from data.dataset import SynthTextDataLoader

from collections import OrderedDict
from craft import CRAFT
from loss.mseloss import Maploss
from torch.autograd import Variable
from utils.util import save_parser, make_logger, AverageMeter
from utils import config
from eval import main
from metrics.eval_det_iou import DetectionIoUEvaluator
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CRAFT new-backtime92')
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser.add_argument('--results_dir', default='/data/workspace/woans0104/CRAFT-re-backtime92/exp/weekly_back_2', type=str,
                    help='Path to save checkpoints')
parser.add_argument('--synthData_dir', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument("--ckpt_path", default='', type=str,
                    help="path to pretrained model")
parser.add_argument('--batch_size', default=16, type = int,
                    help='batch size of training')
parser.add_argument('--iter', default=10, type = int,
                    help='batch size of training')
parser.add_argument('--st_iter', default=0, type = int,
                    help='batch size of training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--lr-decay', default=10000, type=int, help='learning rate decay')
parser.add_argument('--gamma', '--gamma', default=0.8, type=float,
                    help='initial gamma')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--aug', default=False, type=str2bool, help='augmentation')
parser.add_argument('--amp', default=False, type=str2bool, help='augmentation')

#for test


parser.add_argument('--trained_model', default='', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=960, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--isTraingDataset', default=False, type=str2bool, help='test for traing or test data')
#parser.add_argument('--test_folder', default='/home/data/ocr/detection/ICDAR2015', type=str, help='folder path to input images')
parser.add_argument('--test_folder', default='/data/ICDAR2013', type=str, help='folder path to input images')



args = parser.parse_args()


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



if __name__ == "__main__":

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    save_parser(args)
    config.AUG = args.aug


    synthData_dir = {"synthtext": args.synthData_dir}
    synthDataLoader = SynthTextDataLoader(target_size=768, data_dir_list=synthData_dir, mode='train')
    tst_charbox, tst_image, tst_imgtxt = synthDataLoader.load_synthtext(mode='test')
    test_data_li = [tst_charbox, tst_image, tst_imgtxt]



    train_loader = torch.utils.data.DataLoader(synthDataLoader,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               pin_memory=True)


    # logger
    trn_logger, val_logger = make_logger(path=args.results_dir)


    craft = CRAFT(pretrained=True, amp=args.amp)

    if args.st_iter != 0:
        print('success craft_load')
        net_param = torch.load(args.ckpt_path)
        try:
            craft.load_state_dict(copyStateDict(net_param['craft']))
        except:
            craft.load_state_dict(copyStateDict(net_param))


    craft = torch.nn.DataParallel(craft).cuda()
    torch.backends.cudnn.benchmark =True

    optimizer = optim.Adam(craft.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    if args.st_iter != 0:
        print('success optim_load')
        optimizer.load_state_dict(copyStateDict(net_param['optimizer']))
        args.st_iter = net_param['optimizer']['state'][0]['step']
        args.lr = net_param['optimizer']['param_groups'][0]['lr']

    # mixed precision
    if args.amp :
        if args.st_iter != 0 :
            scaler = torch.cuda.amp.GradScaler()
            scaler.load_state_dict(copyStateDict(net_param["scaler"]))
        else:
            scaler = torch.cuda.amp.GradScaler()



    criterion = Maploss()

    #logger
    trn_logger, val_logger = make_logger(path=args.results_dir)

    train_step = args.st_iter
    whole_training_step = args.iter
    update_lr_rate_step = 0
    training_lr = args.lr
    loss_value = 0
    batch_time = 0
    losses = AverageMeter()

    start_time = time.time()
    while train_step < whole_training_step:

        for index, (image, region_image, affinity_image, confidence_mask, confidences) in enumerate(tqdm(train_loader)):
            craft.train()
            if train_step>0 and train_step % args.lr_decay==0:
                update_lr_rate_step += 1
                training_lr = adjust_learning_rate(optimizer, args.gamma, update_lr_rate_step, args.lr)


            images = Variable(image).cuda()
            region_image_label = Variable(region_image).cuda()
            affinity_image_label = Variable(affinity_image).cuda()
            confidence_mask_label = Variable(confidence_mask).cuda()

            if args.amp:
                with torch.cuda.amp.autocast():
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)
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


            # if train_step > 0 and train_step%5==0:
            #     mean_loss = loss_value / 5
            #     loss_value = 0
            #     display_batch_time = time.time()
            #     avg_batch_time = batch_time/5
            #     batch_time = 0
            #
            #     #trn_logger.write([train_step, mean_loss])
            #
            #     print("{}, training_step: {}|{}, learning rate: {:.8f}, training_loss: {:.5f}, avg_batch_time: {:.5f}"
            #           .format(time.strftime('%Y-%m-%d:%H:%M:%S',time.localtime(time.time())), train_step,
            #                   whole_training_step, training_lr, mean_loss, avg_batch_time))



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
                if args.amp:
                    metrics = main(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args, evaluator)
                else:
                    metrics = main(args.results_dir + '/CRAFT_clr_' + repr(train_step)+ '.pth', args, evaluator)
                val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])
                losses.reset()
            train_step += 1
            config.ITER +=1

            if train_step >= whole_training_step : break



    #last model save

    last_time = time.time() - start_time
    print("start time : {}".format(str(start_time)))
    print("present time : {}".format(str(time.time())))
    print("end time : {}".format(str(last_time)))

    f = open(args.results_dir + '/total_time' + '.txt', "w")
    f.write("start time : {}".format(str(start_time)))
    f.write("present time : {}".format(str(time.time())))
    f.write("end time : {}".format(str(last_time)))
    f.close()


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
    if args.amp:
        metrics = main(args.results_dir + '/CRAFT_clr_amp_' + repr(train_step) + '.pth', args, evaluator)
    else:
        metrics = main(args.results_dir + '/CRAFT_clr_' + repr(train_step)+ '.pth', args, evaluator)

    val_logger.write([train_step, losses.avg, str(np.round(metrics['hmean'], 3))])
