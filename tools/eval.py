from __future__ import absolute_import

import os

from got10k.experiments import *

import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"




if __name__ == '__main__':

    # #report_files = ['reports/GOT-10k/performance_25_entries.json']
    # tracker_names = ['UDT_Color128']

    # # setup experiment and plot curves
    # #experiment = ExperimentGOT10k('data/GOT-10k', subset='test')
    # experiment = ExperimentTColor128('testdata/Temple-color-128',result_dir='tcolor',report_dir='tcolor/report')
    # experiment.plot_curves(tracker_names)



    # 指定结果文件路径和跟踪器名称
    result_files = ['tcolor']  # 将路径替换为您的结果文件路径
    tracker_names = ['UDT_Color128']  # 指定跟踪器的名称

    # 设置实验并绘制曲线
    experiment = ExperimentTColor128('testdata/Temple-color-128',result_dir='tcolor/UDT_Color128')  # 将路径替换为TColor128数据集的路径
    experiment.plot_curves(tracker_names)

