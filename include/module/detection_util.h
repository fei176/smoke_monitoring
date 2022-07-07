#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class DetectionHelper {
	static std::vector<cv::Scalar> color;
	static std::vector<std::string> coco_name_eighty;
	static std::vector<std::string> coco_name_ninety_one;
public:
	static std::string& get_name_coco(int i, int mode);
	static cv::Scalar& get_color(int i);
};

class Box {
public:
	Box();
	Box(float x1_, float y1_, float x2_, float y2_, float conf_, float label_);
	float iou(Box& bbox2);
	void resize2img(float ratio, int up, int left);
	void resize2ori(int h, int w);
	cv::Rect rect();
	cv::Point text();
	cv::Rect text_rect();
	float w();
	float h();
	float x1;
	float y1, x2, y2, conf, label;
};