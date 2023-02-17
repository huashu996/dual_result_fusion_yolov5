#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : make_txt.py
@Time    : 2020/11/25 16:51
@desc	 : 划分数据集 ——
            生成：trainval.txt，train.txt，val.txt，test.txt
            为ImageSets文件夹下面Main子文件夹中的训练集+验证集、训练集、验证集、测试集的划分
'''

import os
import random


def ds_partition(annotation_filepath, ds_divide_save__path):
    """
    数据集划分：训练集，验证集，测试集，在ImageSets/Main/生成 train.txt，val.txt，trainval.txt，test.txt
    :param annotation_filepath: 标注文件的路径 Annotations
    :param ds_divide_save__path:   数据集划分保存的路径 ImageSets/Main/
    :return:
    """
    if not os.path.exists(ds_divide_save__path):
        os.mkdir(ds_divide_save__path)

    # train_percent：训练集占（训练集+验证集）的比例
    train_percent = 0.8

    # trainval_percent：（训练集+验证集）占总数据集的比例。测试集所占比例为：1-trainval_percent
    trainval_percent = 1

    temp_xml = os.listdir(annotation_filepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(ds_divide_save__path, 'trainval.txt'), 'w')
    ftest = open(os.path.join(ds_divide_save__path, 'test.txt'), 'w')
    ftrain = open(os.path.join(ds_divide_save__path, 'train.txt'), 'w')
    fval = open(os.path.join(ds_divide_save__path, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    # 自定义数据集文件夹的名称
    dataset_name = r'./drone/'
    annotation_filepath = dataset_name + 'Annotations/'
    divide_save_path = dataset_name + 'ImageSets/Main/'

    ds_partition(annotation_filepath, divide_save_path)
