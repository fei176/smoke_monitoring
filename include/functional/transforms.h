#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace transforms {

	typedef enum clsss {
		CHW = 0,
		HWC = 1
	} DataFormat;

	// use this to normalize the input mat
	cv::Mat normalize(const cv::Mat& mat, const std::vector<float>& mean, const std::vector<float>& std);

	//use this, you need to make sure the size of mean data equels sour.channels(),other wise...
	void normalize(const cv::Mat& sour, cv::Mat& out, const float * mean, const float * std);

	// use this to normalize the input with mean and std.
	void normalize(cv::Mat& mat, const float mean, const float std);

	cv::Size get_new_size(cv::Mat& img, int max_size, float* ratio_value);

	//design for classify
	cv::Mat resize(cv::Mat& mat, bool forced, int w, int h, bool center = true,cv::Scalar pad_value = cv::Scalar::all(0.447));

	//design for detection, no left up padding
	cv::Mat resize(cv::Mat& mat, bool forced, int w, int h, float* ratio, bool center = true, cv::Scalar pad_value = cv::Scalar::all(0.447));

	//design for detection, left up padding
	cv::Mat resize(cv::Mat& mat, bool forced, int w, int h, int* u_v, int* l_v, float* ratio,bool center = true,cv::Scalar pad_value= cv::Scalar::all(0.447));

	Ort::Value make_tensor(const cv::Mat& image, const std::vector<int64_t>& shape,
		const Ort::MemoryInfo& memory_info,
		std::vector<float>& tensor_data,
		const DataFormat& data_format);

	template<typename T>
	Ort::Value make_scaler_tensor(T t, const Ort::MemoryInfo& memory_info) {
		int64_t shape = 1;
		return  Ort::Value::CreateTensor<T>(memory_info, &t, 1, &shape, 1);
	}

	//»­¿òÌí¼ÓÎÄ×Ö¡£
}