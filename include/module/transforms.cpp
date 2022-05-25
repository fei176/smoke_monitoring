#include "transforms.h"

void transforms::normalize(const cv::Mat& sour, cv::Mat& out, const float* mean, const float* std) {
	std::vector<cv::Mat> image_channels;
	cv::split(sour, image_channels);
	for (int i{ 0 }; i < image_channels.size(); i++) {
		transforms::normalize(image_channels[i], mean[i], std[i]);
	}
	cv::merge(image_channels, out);
}

void transforms::normalize(cv::Mat& mat, const float mean, const float std) {
	mat -= mean;
	mat /= std;
}

bool transforms::check_data(const cv::Mat& image, const std::vector<int64_t> shape,
	const DataFormat& data_format) {
	unsigned int h = image.rows;
	unsigned int w = image.cols;
	unsigned int c = image.channels();

	int target_c, target_h, target_w;

	if (data_format == DataFormat::CHW) {
		target_c = shape.at(1);
		target_h = shape.at(2);
		target_w = shape.at(3);
	}
	else{
		target_c = shape.at(3);
		target_h = shape.at(1);
		target_w = shape.at(2);
	}
	return (target_c == -1 || h == target_h) && (target_w == -1 || w == target_w) && c == target_c;
}

Ort::Value  transforms::to_tensor(const cv::Mat& image, const std::vector<int64_t> shape,
	const Ort::MemoryInfo& memory_info,
	std::vector<float>& tensor_data, const DataFormat& data_format) {
	unsigned int h = image.rows;
	unsigned int w = image.cols;
	unsigned int c = image.channels();
	if (data_format == DataFormat::CHW) {
		std::vector<cv::Mat> image_channels;
		cv::split(image, image_channels);
		// CXHXW
		for (unsigned int i = 0; i < c; i++) {
			std::memcpy(tensor_data.data() + i * h *  w, image_channels.at(i).data, h * w * sizeof(float));
		}
	}
	else {
		std::memcpy(tensor_data.data(), image.data, h * w * c * sizeof(float));
	}
	return Ort::Value::CreateTensor<float>(memory_info, tensor_data.data(), h * w * c, shape.data(), shape.size());
}
/// <param name="image">input image</param>
/// <param name="shape">target shape</param>
/// <param name="memory_info">target memory info</param>
/// <param name="tensor_data">tensor source data</param>
/// <param name="data_format">data format</param>
/// <returns></returns>
Ort::Value transforms::make_tensor(const cv::Mat& image, const std::vector<int64_t> shape,
	const Ort::MemoryInfo& memory_info,
	std::vector<float>& tensor_data,
	const DataFormat& data_format) throw (std::runtime_error) {
	if (!transforms::check_data(image, shape, data_format)) {
		//TODO 写一个从vector 构造一个字符串的方法，好提升报错信息
		throw std::runtime_error("input image is dismatcded with target shape");
	}
	return transforms::to_tensor(image, shape, memory_info, tensor_data, data_format);
}