#include "functional/transforms.h"
#include "onnx/onnxutils.h"

cv::Mat transforms::normalize(const cv::Mat& mat, const std::vector<float>& mean, const std::vector<float>& std) {
	cv::Mat image;
	if ((mean.size() != 1 || std.size() != 1) && (mean.size() != mat.channels() || mean.size() != std.size())) {
		throw std::runtime_error("mean (std) shape don't match the image channels");
	}
	std::vector<float> mean_ = mean;
	std::vector<float> std_ = std;
	if (mean.size() == 1) {
		for (int i = 1; i < mat.channels(); i++) {
			mean_.push_back(mean[0]);
			std_.push_back(mean[0]);
		}
	}
	transforms::normalize(mat, image, mean_.data(), std_.data());
	return image;
}

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

cv::Size transforms::get_new_size(cv::Mat& img, int max_size,float * ratio_value) {
	int h = img.rows;
	int w = img.cols;
	float ratio = std::min(max_size * 1.0 / h, max_size * 1.0 / w);
	ratio = std::min(ratio, float(1.0));
	*ratio_value = ratio;
	return cv::Size(int(w * ratio), int(h * ratio));
}

cv::Mat transforms::resize(cv::Mat& mat, bool forced, int new_w, int new_h, bool center,cv::Scalar pad_value) {
	int u, l;
	float ratio;
	return resize(mat, forced, new_w, new_w, &u, &l,&ratio, center, pad_value);
}

cv::Mat transforms::resize(cv::Mat& mat, bool forced, int new_w, int new_h, float* ratio, bool center, cv::Scalar pad_value) {
	int u, l;
	return resize(mat, forced, new_w, new_w, &u, &l, ratio, center, pad_value);
}

/// <summary>
/// 将输入图像（长边）resize到不大于target的最接近32倍数的大小，随后对短边补零 if forced=false;
/// 强行resize到hw大小，if forced=true.
/// </summary>
/// <param name="mat"></param>
/// <param name="forced"></param>
/// <param name="target"></param>
/// <returns></returns>
cv::Mat transforms::resize(cv::Mat& mat, bool forced, int new_w,int new_h,int * u_v,int * l_v, float * ratio_value,bool center,cv::Scalar pad_value) {
	cv::Mat resize_img;
	int h = mat.rows;
	int w = mat.cols;
	if (forced) {
		// if force 
		if (h != new_h && w != new_w) {
			cv::resize(mat, resize_img, cv::Size(new_w, new_h));
			return resize_img;
		}
		return mat;
	}
	float ratio = std::min(new_w * 1.0 / w, new_h * 1.0 / h);
	ratio = std::min(float(1.0), ratio);
	*ratio_value = ratio;
	int temp_h = int(std::round(ratio * h));
	int temp_w = int(std:: round(ratio * w));
	if (temp_h != h && temp_w != w) {
		cv::resize(mat, resize_img, cv::Size(temp_w, temp_h));
	}
	int u=0, d=0, l=0, r = 0;
	if (center) {
		u = (new_h - temp_h) / 2;
		d = new_h - temp_h - u;
		l = (new_w - temp_w) / 2;
		r = new_w - temp_w - l;
	}
	else {
		d = new_h - temp_h;
		r = new_w - temp_w;
	}
	cv::Mat pad_mat;
	cv::copyMakeBorder(resize_img, pad_mat, u, d, l, r, cv::BorderTypes::BORDER_CONSTANT, pad_value);
	*u_v = u;
	*l_v = l;
	return pad_mat;
}

/// <summary>
/// 根据cv::Mat构造一个tensor，输入数据形状，mem_info,一个buffer[还不是模板]，tensor format
/// </summary>
/// <param name="image">input image</param>
/// <param name="shape">target shape</param>
/// <param name="memory_info">target memory info</param>
/// <param name="tensor_data">tensor source data</param>
/// <param name="data_format">data format</param>
/// <returns>Ort::Value</returns>
Ort::Value transforms::make_tensor(const cv::Mat& image, const std::vector<int64_t>& shape,
	const Ort::MemoryInfo& memory_info,
	std::vector<float>& tensor_data,
	const DataFormat& data_format) throw (std::runtime_error) {
	unsigned int h = image.rows;
	unsigned int w = image.cols;
	unsigned int c = image.channels();
	if (data_format == DataFormat::CHW) {
		std::vector<cv::Mat> image_channels;
		cv::split(image, image_channels);
		// CXHXW
		for (unsigned int i = 0; i < c; i++) {
			std::memcpy(tensor_data.data() + i * h * w, image_channels.at(i).data, h * w * sizeof(float));
		}
	}
	else {
		std::memcpy(tensor_data.data(), image.data, h * w * c * sizeof(float));
	}
	return Ort::Value::CreateTensor<float>(memory_info, tensor_data.data(), h * w * c, shape.data(), shape.size());
}