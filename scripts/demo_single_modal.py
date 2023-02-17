# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading
import argparse

from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolov5_detector import Yolov5Detector, draw_predictions
from mono_estimator import MonoEstimator
from functions import get_stamp, publish_image
from functions import display, print_info

parser = argparse.ArgumentParser(
    description='Demo script for dual modal peception')
    
parser.add_argument('--print', action='store_true',
    help='Whether to print and record infos.')
    
parser.add_argument('--sub_image', default='/pub_rgb', type=str,
    help='The image topic to subscribe.')
    
parser.add_argument('--pub_image', default='/result', type=str,
    help='The image topic to publish.')
    
parser.add_argument('--calib_file', default='../conf/calibration_image.yaml', type=str,
    help='The calibration file of the camera.')
    
parser.add_argument('--modality', default='RGB', type=str,
    help='The modality to use. This should be `RGB` or `T`.')
    
parser.add_argument('--indoor', action='store_true',
    help='Whether to use INDOOR detection mode.')
    
parser.add_argument('--frame_rate', default=10, type=int,
    help='Working frequency.')
    
parser.add_argument('--display', action='store_true',
    help='Whether to display and save all videos.')
    
args = parser.parse_args()

image_lock = threading.Lock() # 在多线程中使用lock可以让多个线程在共享资源的时候不会“乱”

def image_callback(image):
    global image_stamp, image_frame
    image_lock.acquire()
    image_stamp = get_stamp(image.header)
    image_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image_lock.release()

def timer_callback(event):
    global image_stamp, image_frame
    image_lock.acquire()
    cur_stamp = image_stamp
    cur_frame = image_frame.copy()
    image_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    if args.indoor:
        labels, scores, boxes = detector.run(
            cur_frame, conf_thres=0.50, classes=[0]
        ) # person
    else:
        labels, scores, boxes = detector.run(
            cur_frame, conf_thres=0.50, classes=[0, 1, 2, 3, 4]
        ) # pedestrian, cyclist, car, bus, truck
    labels_temp = labels.copy()
    labels = []
    for i in labels_temp:
        labels.append(i if i not in ['pedestrian', 'cyclist'] else 'person')
    
    locations = mono.estimate(boxes)
    indices = [i for i in range(len(locations)) if locations[i][1] > 0 and locations[i][1] < 200]
    labels, scores, boxes, locations = \
        np.array(labels)[indices], np.array(scores)[indices], boxes[indices], np.array(locations)[indices]
    distances = [(loc[0] ** 2 + loc[1] ** 2) ** 0.5 for loc in locations]
    cur_frame = cur_frame[:, :, ::-1].copy() # to BGR
    for i in reversed(np.argsort(distances)):
        cur_frame = draw_predictions(
            cur_frame, str(labels[i]), float(scores[i]), boxes[i], location=locations[i]
        )
    
    if args.display:
        if not display(cur_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")
    cur_frame = cur_frame[:, :, ::-1] # to RGB
    publish_image(pub, cur_frame)
    delay = round(time.time() - start, 3)
    
    if args.print:
        print_info(frame, cur_stamp, delay, labels, scores, boxes, locations, file_name)

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("single_modal_perception", anonymous=True, disable_signals=True)
    frame = 0
    
    # 记录时间戳和检测结果
    if args.print: #判断命令行中有没有print
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    # 设置标定参数
    if not os.path.exists(args.calib_file):
        raise ValueError("%s Not Found" % (args.calib_file))
    mono = MonoEstimator(args.calib_file, print_info=args.print)
    
    # 初始化Yolov5Detector
    if args.indoor: #如果在命令行中输入 python3 demo_dual_modal.py --indoor 则会运行下面代码
        detector = Yolov5Detector(weights='weights/coco/yolov5s.pt')
    else:
        if args.modality.lower() == 'rgb':
            detector = Yolov5Detector(weights='weights/seumm_visible/yolov5s_100ep_pretrained.pt')
        elif args.modality.lower() == 't':
            detector = Yolov5Detector(weights='weights/seumm_lwir/yolov5s_100ep_pretrained.pt')
        else:
            raise ValueError("The modality must be `RGB` or `T`.")
    
    # 准备图像序列
    image_stamp = None
    image_frame = None
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1,
        buff_size=52428800)
    while image_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s...' % args.sub_image)
    print('  Done.\n')
    
    # 保存视频
    if args.display:
        win_h, win_w = image_frame.shape[0], image_frame.shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)
    
    # 启动定时检测线程
    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)
    
    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
