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

	cv::Mat resize(cv::Mat& mat, bool forced, int w, int h, bool center = true,cv::Scalar pad_value = cv::Scalar::all(0.447));

	cv::Mat resize(cv::Mat& mat, bool forced, int w, int h, float* ratio, bool center = true, cv::Scalar pad_value = cv::Scalar::all(0.447));

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