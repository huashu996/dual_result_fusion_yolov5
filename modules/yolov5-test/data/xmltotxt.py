#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : voc2yolo_label.py
@Time    : 2020/11/25 16:54
@desc	 : 类别文件格式转换 ——
            把数据集标注格式转换成yolo_txt格式，即将每个xml标注提取bbox信息为txt格式
'''

import xml.etree.ElementTree as ET
import os

sets = ['train', 'test', 'val']

# 改成自己训练所需要的类
classes = ['cell phone']


def convert(size, box):
    """
    将 VOC 的标注转为 yolo的标注，即 xyxy -> xywh
    :param size: 图片尺寸
    :param box: 标注框（xyxy）
    :return:
    """
    dw = 1. / size[0]
    dh = 1. / size[1]

    # 中心点坐标
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    # 宽高
    w = box[1] - box[0]
    h = box[3] - box[2]

    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(annotation_filepath, label_dir, image_id):
    """
    VOC 标注的结果转为 yolo 数据集的标注结果
    :param annotation_filepath: VOC标注文件夹路径
    :param label_dir: 解析成 yolo 标注文件夹的路径
    :param image_id: 文件名
    :return:
    """
    in_file = open(annotation_filepath + '%s.xml' % (image_id))
    out_file = open(label_dir + '%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':

    # 当前文件的绝对路径
    abs_path = os.getcwd() + '/'

    # 自定义数据集文件夹的名称
    dataset_name = 'drone'

    # VOC数据集标注文件
    annotation_filepath = dataset_name + '/Annotations/'

    # 数据集划分文件
    txtfile_dir = dataset_name + '/ImageSets/Main/'

    # coco训练图像输入的文件夹
    image_dir = dataset_name + '/images/'

    # 标注文件解析之后的文件夹路径
    label_dir = dataset_name + '/labels/'

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for image_set in sets:

        txtfile = txtfile_dir + '%s.txt' % (image_set)

        image_ids = open(txtfile).read().strip().split()
        list_file = open(dataset_name + '/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write((abs_path + image_dir + '%s.jpg\n' % (image_id)).replace('\\', '/'))
            convert_annotation(annotation_filepath, label_dir, image_id)
        list_file.close()
