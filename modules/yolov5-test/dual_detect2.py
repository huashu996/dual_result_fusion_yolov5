#! /usr/bin/env python2
 
from __future__ import print_function
 
#python2
import os
import roslib
import rospy
from std_msgs.msg import Header
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from ros_numpy import msgify
#from cv_bridge import CvBridge
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import threading
import time
import numpy as np
import cv2
import threading
#python3
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import time
 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
 
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
@torch.no_grad()
 
class SubscribeAndPublish:
    def __init__(self):
        self.all_obstacle_str=''
 
        #self.sub1_name="/cam_rgb/usb_cam/image_rect_color"
        self.sub1_name="/pub_rgb"
        self.sub1= rospy.Subscriber(self.sub1_name, Image,self.callback_rgb)
        self.sub2_name="/pub_t"
        self.sub2= rospy.Subscriber(self.sub2_name, Image,self.callback_t)
        
        
 
        self.pub1_name="detect_rgb"
        self.pub1= rospy.Publisher(self.pub1_name, Image,queue_size=1)
        self.pub2_name="detect_t"
        self.pub2= rospy.Publisher(self.pub2_name, Image,queue_size=1)
        self.timed=0
        self.model=model
        self.device=device
        self.stride=32 
        self.names=['pedestrian', 'cyclist', 'car', 'bus', 'truck', 'traffic_light', 'traffic_sign']
        self.pt=True
        self.jit=False
        self.onnx=False
        self.engine=False
        self.img_rgb=[]
        #self.mode=0
        #self.path_rgb='./img/img_rgb'
        #self.path_t='./img/img_t'
        #self.path_info='./img/img_info'
        # self.bridge = CvBridge()
 
       
    def callback_rgb(self,data):
        print('callback1')
        img_rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        img_rgb=img_rgb[:,:,::-1]
        self.img_rgb=img_rgb
        cv2.imwrite('./temp/rgb/rgb.jpg',img_rgb)
        
 
     
    def callback_t(self,data):
        print('callback2')
        time_run1=time.time()
        timed=time.time()
        print('time:',timed-self.timed)
        self.timed=timed
        
        img_t = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        img_t=img_t[:,:,::-1]
        cv2.imwrite('./temp/t/t.jpg',img_t)
        print('save image successful!')
        #check_requirements(exclude=('tensorboard', 'thop'))
 
        img_t,img_rgb=self.run(**vars(opt))
        
        
        #cv2.imshow('i',img)
        #cv2.waitKey(1)
        #print(len(img))
        #print('detection!')
        #self.pub1.publish(CvBridge().cv2_to_imgmsg(img_rgb,"rgb8"))
        if len(img_t)>0:
           print('send img')
           self.publish_image(self.pub2,img_t,'base_link')
           self.publish_image(self.pub1,img_rgb,'base_link')
           time_run2=time.time()
           time_run=time_run2-time_run1
           print('time_run:',time_run)
           print('')
    
    def publish_image(self,pub, data, frame_id='base_link'):
        assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
        header = Header(stamp=rospy.Time.now())
        header.frame_id = frame_id
    
        msg = Image()
        msg.height = data.shape[0]
        msg.width = data.shape[1]
        msg.encoding = 'rgb8'
        msg.data = np.array(data).tostring()
        msg.header = header
        msg.step = msg.width * 1 * 3
    
        pub.publish(msg)
    
    def get_time_stamp(self):
        ct = time.time()
        local_time = time.localtime(ct)
        data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        data_secs = (ct - int(ct)) * 1000
        time_stamp = "%s.%03d" % (data_head, data_secs)
        #print(time_stamp)
        stamp = ("".join(time_stamp.split()[0].split("-"))+"".join(time_stamp.split()[1].split(":"))).replace('.', '')
        #print(stamp)
        return stamp
    
    def run(self,weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'temp',  # file/dir/URL/glob, 0 for webcam
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
 
        # Directories
        #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Load model
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size
 
        # Half
 
        
        # Dataloader
        '''if webcam:
            print('webcam')
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            print('no')
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size'''
        dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=self.pt)
        rgb_data=LoadImages('./temp/rgb', img_size=imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        
        # Run inference
        #self.model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in rgb_data:
            im=torch.from_numpy(im).to(self.device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            rgb=im
            rgb0s=im0s
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
 
            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            
            pred = self.model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
 
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
 
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
 
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                '''if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)'''
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                rgb0=rgb0s.copy()
                p = Path(p)  # to Path
                #save_path = str(save_dir / p.name)  # im.jpg
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                rgbc = rgb0.copy() if save_crop else rgb0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                annotator_rgb = Annotator(rgb0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
 
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
 
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
 
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            annotator_rgb.box_label(xyxy, label, color=colors(c, True))
                            '''if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)'''
 
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
 
                # Stream results
                im0 = annotator.result()
                rgb0=annotator_rgb.result()
                
        return im0,rgb0
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './runs/train/exp2/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'temp/t', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt
 
 
 
def main(opt,model,device):
    rospy.init_node('biaoding_ws', anonymous=True)
 
    #####################
    t=SubscribeAndPublish()
    #####################
    rospy.spin()
if __name__ == "__main__":
    opt = parse_opt()
    device = ''
    weights = './runs/train/exp2/weights/best.pt'
    dnn=False
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    main(opt,model,device)
