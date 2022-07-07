# smoke_monitoring
**Final goal:**

classify and detect smoke、deploying model with onnx 

use c++ 11 features、threadpool and http in this project in order to make this project usefull.



**Intermediate target：**

Implement some classic and start of art networks.

**1、Classify:**

This type of network is the easiest to implement, so we just get the onnx model,call the forward,then it's over.

Implemented classic network：resnet、inception、desen set、ghost net、resnetxt with ibn、mobile net、sequeeze net、shuffle net、

**2、Detection:**

ori_pic、Yolo、Yolox、Detr

<img src="./result_img/dog.jpg" alt="dog" style="zoom:30%;" /><img src="./result_img/dog_pred_yolov5.jpg" alt="dog_pred" style="zoom:30%;" /><img src="./result_img/dog_pred_yolox.jpg" alt="dog_pred" style="zoom:30%;" /><img src="./result_img/dog_pred_detr.jpg" alt="dog_pred" style="zoom:30%;" />

onnx_files:

[Yolov5](链接：https://pan.baidu.com/s/1HunafZ8AGq-D5IiE1EnYwg?pwd=1234 提取码：1234) [YoloX](链接：https://pan.baidu.com/s/1CYF-l-WTHcLE_CoC42JVkg?pwd=1234 提取码：1234) [Detr](链接：https://pan.baidu.com/s/1Wahnw0sTb28xRUI558W4PQ?pwd=1234 提取码：1234) [SSD]((链接：https://pan.baidu.com/s/1Wahnw0sTb28xRUI558W4PQ?pwd=1234 提取码：1234) ) [Fcos](链接：https://pan.baidu.com/s/1iphxv0JM45fedMNG-snQQg?pwd=5xin 提取码：5xin) [yolov6](链接：https://pan.baidu.com/s/1prpFRURdyROI3P6fCB2wsg?pwd=cjmn 提取码：cjmn)

still working...



**How to use:**

quick use：http://120.48.25.3:6573

opencv is required

set boost include dir, onnx include dir and onnx lib dir in CMakeLists.txt, than use cmake and make to compile the project, the output file is in ./build

how to use: 

1、./smoke 127.0.0.1 1234 /home/path/to/web/dir /home/path/to/weights/dir

2、open 127.0.0.1:1234 in browser

3、choose a model，fill out the parameters，it will show the detection result after 0-1s (relying on cpu performance ) in most cases(i hope so,😅）

 <img src="./result_img/temp.png" alt="dog" style="zoom:0%;" />

**Feature work：**

GPU-supported, more models....
