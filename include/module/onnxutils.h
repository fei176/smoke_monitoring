#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

namespace onnxutils {
	typedef enum clsss {
		success,
		bad_count,
		bad_dims,
		bad_shape
	} DataError;

	std::string print_shape(std::vector<int64_t>& shape);

	DataError check_input(const std::vector<int64_t>& input, const std::vector<int64_t>& target);

	DataError check_input(const std::vector<std::vector<int64_t>>& input,const std::vector<std::vector<int64_t>>& target);

	std::string get_error_info(std::vector<std::vector<int64_t>>& input, std::vector<std::vector<int64_t>>& target);

	class ModelIOInfo {
	public:
		ModelIOInfo();
		ModelIOInfo(Ort::Session* session, Ort::AllocatorWithDefaultOptions* allocator);
		ModelIOInfo& operator=(onnxutils::ModelIOInfo&&);
		virtual ~ModelIOInfo();
		std::vector<std::vector<int64_t>>& get_input_shapes();
		std::vector<char*>& get_input_names();
		std::vector<ONNXTensorElementDataType>& get_input_data_format();
		size_t get_input_count();

		std::vector<std::vector<int64_t>>& get_ouput_shapes();
		std::vector<char*>& get_out_names();
		std::vector<ONNXTensorElementDataType>& get_output_data_format();
		size_t get_output_count();

		const std::string& print_info();
	private:
		std::vector<std::vector<int64_t>> input_shapes;
		std::vector<char*> input_names;
		std::vector<ONNXTensorElementDataType> input_data_format;
		size_t input_count;

		std::vector<std::vector<int64_t>> output_shapes;
		std::vector<char*> output_names;
		std::vector<ONNXTensorElementDataType> output_data_format;
		size_t output_count;

		std::string info;
		Ort::AllocatorWithDefaultOptions* allocator;
	};
}