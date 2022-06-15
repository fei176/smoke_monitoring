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

still working...

**3、segmentation**







How to use:

Because of some restrictions, this project run on a fixed directory(windows, cpu only，onnx runtime), for now,it's c:/project

1、unzip the "project.zip" to c:/project

2、cd to c:/project

3、run mnist.exe , default listening on port 127.0.0.1:6573

4、open 127.0.0.1:6573 in browser

5、choose a model，fill out the parameters，it will show the detection result after 0-1s (relying on cpu performance ) in most cases(i hope so,😅）![image-20220615230141944](.\result_img\temp.png)

 

**Feature work：**

fixed bugs，GPU-supported, more model, write makefile.txt for cross platform supported...
