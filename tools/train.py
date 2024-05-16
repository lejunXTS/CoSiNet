from __future__ import absolute_import

import os

import torch
from got10k.datasets import *

import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

from siamfc.siamfc1 import TrackerSiamFC

#因为作者使用了GOT-10k这个工具箱，train.py代码非常少
#首先我们就需要按照GOT-10k download界面去下载好数据集，并且按照这样的文件结构放好
#（因为现在用不到验证集和测试集，可以先不用下，训练集也只要先下载1个split，所以就需要把list.txt中只保留前500项，
#因为GOT-10k_Train_000001里面有500个squences）

if __name__ == '__main__':
    root_dir = os.path.expanduser('/home/asus/ly/SiamDUL/GOT-10k')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    net_path = None
    tracker = TrackerSiamFC(net_path)
    tracker.train_over(seqs)
