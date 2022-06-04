#pragma once
#include <algorithm>
#include <opencv2/opencv.hpp>
class DetectionHelper {
	static std::vector<cv::Scalar> color;
	static std::vector<std::string> coco_name;
public:
	static std::string& get_name(int i);
	static cv::Scalar& get_color(int i);
};

class Box {
public:
	Box();
	Box(float x1_, float y1_, float x2_, float y2_, float conf_, float label_);
	float iou(Box& bbox2);
	void resize2img(float ratio, int up, int left);
	cv::Rect rect();
	cv::Point text();
	float w();
	float h();
	float x1, y1, x2, y2, conf,label;
};
