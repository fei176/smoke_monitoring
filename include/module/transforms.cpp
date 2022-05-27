#include "module/transforms.h"
#include "module/onnxutils.h"


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

cv::Mat transforms::resize(cv::Mat& mat, bool forced, int target) {
	cv::Mat resize_img;
	if (forced) {
		cv::resize(mat, resize_img, cv::Size(target, target));
		return resize_img;
	}
	int w = mat.cols;
	int h = mat.rows;
	int max_size = std::max(w, h);
	int new_size = std::max(int(target / 32),1) * 32;
	int new_size_two = 0;
	int u=0, d=0, l=0, r = 0;
	if (max_size == w) {
		new_size_two = h / (w * 1.0 / new_size);
		w = new_size;
		h = new_size_two;
		u = (w - h) / 2;
		d = w - h - u;
	}
	else {
		new_size_two = w / (h * 1.0 / new_size);
		h = new_size;
		w = new_size_two;
		l = (h - w) / 2;
		r = h - w - l;
	}
	cv::resize(mat, resize_img, cv::Size(w, h));
	cv::Mat pad_mat;
	cv::copyMakeBorder(resize_img, pad_mat, u, d, l, r, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0.));
	return pad_mat;
}

bool transforms::check_data(const std::vector<int64_t> input, const std::vector<int64_t>& target) {
	return onnxutils::check_input(input, target) == onnxutils::DataError::success;
}

bool transforms::check_data(const cv::Mat& image, const std::vector<int64_t>& shape,
	const DataFormat& data_format) {
	unsigned int h = image.rows;
	unsigned int w = image.cols;
	unsigned int c = image.channels();

	std::vector<int64_t> input_shape;
	if (data_format == DataFormat::CHW) {
		input_shape = { 1, c,h,w };
	}
	else {
		input_shape = { 1,h,w,c};
	}
	return onnxutils::check_input(input_shape, shape) == onnxutils::DataError::success;
}

Ort::Value  transforms::to_tensor(const cv::Mat& image, const std::vector<int64_t>& shape,
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
Ort::Value transforms::make_tensor(const cv::Mat& image, const std::vector<int64_t>& shape,
	const Ort::MemoryInfo& memory_info,
	std::vector<float>& tensor_data,
	const DataFormat& data_format) throw (std::runtime_error) {
	if (!transforms::check_data(image, shape, data_format)) {
		//TODO 写一个从vector 构造一个字符串的方法，好提升报错信息
		throw std::runtime_error("input image is dismatcded with target shape");
	}
	return transforms::to_tensor(image, shape, memory_info, tensor_data, data_format);
}