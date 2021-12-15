import os
import cv2
import numpy as np
import time

import torch
from torch.autograd import Variable

#from utils import craft_utils
from data import imgproc
from collections import Iterable

def save_parser(args):

    """ final options """
    with open(f'{args.results_dir}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)


def make_logger(path=False):

    # mode = iter or epoch

    def logger_path(path):

        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')

        trn_logger_path = os.path.join(f'{path}', f'train.log')
        val_logger_path = os.path.join(f'{path}', f'validation.log')


        return trn_logger_path, val_logger_path


    trn_logger_path, val_logger_path = logger_path( path)

    trn_logger = Logger(trn_logger_path)
    val_logger = Logger(val_logger_path)

    return trn_logger, val_logger


def split_logger(lang_dict):
    eopch_li = []
    loss_li = []
    #acc_li = []
    #ned_li = []

    for i in lang_dict:
        new_dict = i.split()
        eopch_li.append(int(new_dict[0]))
        loss_li.append(float(new_dict[1]))
        #acc_li.append(float(new_dict[2]))
        #ned_li.append(float(new_dict[3]))

    return eopch_li, loss_li

def read_txt(path):
    with open(path, 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]

    eopch_li, loss_li = split_logger(lang_dict)

    return eopch_li, loss_li

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log





class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)

        else:
            pass