#include "module/detection_util.h"

std::vector<cv::Scalar>DetectionHelper::color = std::vector<cv::Scalar>{ cv::Scalar(221,221,221),
cv::Scalar(170,170,170),cv::Scalar(136,136,136),cv::Scalar(102,102,102),cv::Scalar(68,68,68),
cv::Scalar(0,0,0),cv::Scalar(255,183,221),cv::Scalar(255,136,194),cv::Scalar(255,68,170),
cv::Scalar(255,0,136),cv::Scalar(193,0,102),cv::Scalar(162,0,85),cv::Scalar(140,0,68),
cv::Scalar(255,204,204),cv::Scalar(255,136,136),cv::Scalar(255,51,51),cv::Scalar(255,0,0),
cv::Scalar(204,0,0),cv::Scalar(170,0,0),cv::Scalar(136,0,0),cv::Scalar(255,200,180),
cv::Scalar(255,164,136),cv::Scalar(255,119,68),cv::Scalar(255,85,17),cv::Scalar(230,63,0),
cv::Scalar(198,51,0),cv::Scalar(164,45,0), cv::Scalar(255,221,170),cv::Scalar(255,187,102),
cv::Scalar(255,170,51),cv::Scalar(255,136,0),cv::Scalar(238,119,0),cv::Scalar(204,102,0),
cv::Scalar(187,85,0), cv::Scalar(255,238,153),cv::Scalar(255,221,85),cv::Scalar(255,204,34),
cv::Scalar(255,187,0),cv::Scalar(221,170,0),cv::Scalar(170,119,0),cv::Scalar(136,102,0), cv::Scalar(255,255,187),
cv::Scalar(255,255,119),cv::Scalar(255,255,51),cv::Scalar(255,255,0),cv::Scalar(238,238,0),
cv::Scalar(187,187,0),cv::Scalar(136,136,0), cv::Scalar(238,255,187),
cv::Scalar(221,255,119),cv::Scalar(204,255,51),cv::Scalar(187,255,0),cv::Scalar(153,221,0),
cv::Scalar(136,170,0),cv::Scalar(102,136,0),cv::Scalar(204,255,153),cv::Scalar(187,255,102),cv::Scalar(153,255,51),
cv::Scalar(119,255,0),cv::Scalar(102,221,0),cv::Scalar(85,170,0),cv::Scalar(34,119,0),
cv::Scalar(153,255,153),cv::Scalar(102,255,102),cv::Scalar(51,255,51),cv::Scalar(0,255,0),
cv::Scalar(0,221,0),cv::Scalar(0,170,0),cv::Scalar(0,136,0), cv::Scalar(187,255,238),
cv::Scalar(119,255,204),cv::Scalar(51,255,170),cv::Scalar(0,255,153),cv::Scalar(0,221,119),
cv::Scalar(0,170,85),cv::Scalar(0,136,68), cv::Scalar(170,255,238),
cv::Scalar(119,255,238),cv::Scalar(51,255,221),cv::Scalar(0,255,204),
cv::Scalar(0,221,170),cv::Scalar(0,170,136),cv::Scalar(0,136,102),cv::Scalar(153,255,255),
cv::Scalar(102,255,255),cv::Scalar(51,255,255),cv::Scalar(0,255,255),cv::Scalar(0,221,221),
cv::Scalar(0,170,170),cv::Scalar(0,136,136),cv::Scalar(204,238,255),
cv::Scalar(119,221,255),cv::Scalar(51,204,255),cv::Scalar(0,187,255),cv::Scalar(0,159,204),
cv::Scalar(0,136,168),cv::Scalar(0,119,153),cv::Scalar(204,221,255),
cv::Scalar(153,187,255),cv::Scalar(85,153,255),cv::Scalar(0,102,255),cv::Scalar(0,68,187),
cv::Scalar(0,60,157),cv::Scalar(0,51,119),cv::Scalar(204,204,255),
cv::Scalar(153,153,255),cv::Scalar(85,85,255),cv::Scalar(0,0,255),cv::Scalar(0,0,204),
cv::Scalar(0,0,170),cv::Scalar(0,0,136),cv::Scalar(204,187,255),cv::Scalar(159,136,255),
cv::Scalar(119,68,255),cv::Scalar(85,0,255),cv::Scalar(68,0,204),cv::Scalar(34,0,170),cv::Scalar(34,0,136) };
std::vector<std::string>DetectionHelper::coco_name_eighty = std::vector <std::string>{ "person","bicycle","car",
"motorbike","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
"parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
"backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
"baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
"pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop",
"mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
"clock","vase","scissors","teddy bear","hair drier","toothbrush" };

std::vector<std::string>DetectionHelper::coco_name_ninety_one = std::vector <std::string>{ "person","bicycle","car",
"motorbike","airplane","bus","train","truck","boat","traffic light","fire hydrant","use less","stop sign",
"parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","use less",
"backpack","umbrella","use less","use less","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
"baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","use less","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
"pizza","donut","cake","chair","sofa","pottedplant","bed","use less","diningtable","use less","use less","toilet",
"use less","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
"toaster","sink","refrigerator","use less","book","clock","vase","scissors","teddy bear","hair drier","toothbrush" };


std::string& DetectionHelper::get_name_coco(int i, int mode) {
	return mode == 80 ? coco_name_eighty[i] : coco_name_ninety_one[i];
}

cv::Scalar& DetectionHelper::get_color(int i) {
	return color[i];
}

Box::Box() {
	x1 = y1 = x2 = y2 = 0.0;
	conf = 0.0;
	label = 0.0;
}

// x1 y1 also are x y 
Box::Box(float x1_, float y1_, float x2_, float y2_, float conf_, float label_) {
	x1 = x1_;
	y1 = y1_;
	x2 = x2_;
	y2 = y2_;
	conf = conf_;
	label = label_;
}

float Box::iou(Box& bbox2) {
	float area1 = (x2 - x1) * (y2 - y1);
	float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
	float x1_ = std::max(x1, bbox2.x1);
	float y1_ = std::max(y1, bbox2.y1);
	float x2_ = std::min(x2, bbox2.x2);
	float y2_ = std::min(y2, bbox2.y2);
	float area_union = std::max((y2_ - y1_), float(0.0)) * std::max((x2_ - x1_), float(0.0));
	return area_union / (area1 + area2 - area_union);
}


void Box::resize2img(float ratio, int up, int left) {
	// 0-1(wh) -> ori 0-1(wh)
	x1 -= left;
	y1 -= up;
	x2 -= left;
	y2 -= up;
	x1 /= ratio;
	y1 /= ratio;
	x2 /= ratio;
	y2 /= ratio;
}

void Box::resize2ori(int h, int w) {
	// 0-1 -> wh
	x1 *= w;
	y1 *= h;
	x2 *= w;
	y2 *= h;
}

float Box::h() {
	return y2 - y1;
}

float Box::w() {
	return x2 - x1;
}

cv::Rect Box::rect() {
	return cv::Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1));
}

cv::Point Box::text() {
	return cv::Point(int(x1), int(y1));
}

cv::Rect Box::text_rect() {
	return cv::Rect(int(x1), int(y1 - 20), int(150), int(30));
}
