#pragma once
#include "module/module.h"
#include "module/detection.h"
#include "core/functional.h"
#include <opencv2/opencv.hpp>


class Detection: public Module {
public:
	virtual cv::Mat call(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
		transforms::DataFormat data_format= transforms::DataFormat::CHW) = 0;
	virtual std::vector<Box> call_box(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
		transforms::DataFormat data_format= transforms::DataFormat::CHW) = 0;
	virtual void adjust_par(int h, int w, float nms_thresh, float confidence_thresh) = 0;
	virtual ~Detection();
};
