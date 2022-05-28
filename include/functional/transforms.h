#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace transforms {

	typedef enum clsss {
		CHW = 0,
		HWC = 1
	} DataFormat;

	void normalize(const cv::Mat& sour, cv::Mat& out, const float * mean, const float * std);

	void normalize(cv::Mat& mat, const float mean, const float std);

	cv::Mat resize(cv::Mat& mat, bool forced, int target);

	Ort::Value make_tensor(const cv::Mat& image, const std::vector<int64_t>& shape,
		const Ort::MemoryInfo& memory_info,
		std::vector<float>& tensor_data,
		const DataFormat& data_format);
}