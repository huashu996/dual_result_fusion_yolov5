# dual_modal_perception

ROS package for dual modal perception (rgbt)
## 简介
   此双模态检测是通过对可见光和红外图像分别训练，得到两个weight，在运行时会对两种图像分别检测，最后对检测结果求极大似然，并且能够对目标进行测距。
## 安装
 - 建立ROS工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone git@github.com:huashu996/dual_result_fusion_yolov5.git --recursive
   cd ..
   catkin_make
   ```
 - 使用Anaconda设置环境依赖
   ```Shell
   conda create -n yolov5.v5.0 python=3.8
   conda activate yolov5.v5.0
   cd dual_modal_perception
   pip install -r requirements.txt
   pip install catkin_tools
   pip install rospkg
   ```
 - 准备可见光模态的模型文件，并保存至目录`dual_modal_perception/modules/yolov5-test/weights/seumm_visible/`
 - 准备红外光模态的模型文件，并保存至目录`dual_modal_perception/modules/yolov5-test/weights/seumm_lwir/`
 - 改动demo_dual_modal.py文件中权重的路径
   ```Shell  
        elif args.modality.lower() == 'rgbt': #双模态  
        detector1 = Yolov5Detector(weights='weights/yolov5s.pt')  
        detector2 = Yolov5Detector(weights='weights/yolov5s.pt')  
   ```
## 参数配置
 - 编写相机标定参数`dual_modal_perception/conf/calibration_image.yaml`
   ```
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [859, 0, 339, 0, 0, 864, 212, 0, 0, 0, 1, 0]
   Height: 2.0 # the height of the camera (meter)
   DepressionAngle: 0.0 # the depression angle of the camera (degree)
   ```

## 运行
 - 启动双模态检测算法（检测结果图像可由`rqt_image_view /result`查看）
   ```
   python3 demo_dual_modal.py
   
   # If you want print infos and save videos, run
   python3 demo_dual_modal.py --print --display
   ```
 - 启动单模态检测算法（检测结果图像可由`rqt_image_view /result`查看）
   ```
   python3 demo_single_modal.py
   
   # If you want print infos and save videos, run
   python3 demo_single_modal.py --print --display
   ```

