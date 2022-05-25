#include "onnxutils.h"

std::string onnxutils::print_shape(std::vector<int64_t>& shape) {
	std::string format="shape :[";
	for (auto x : shape) {
		format += std::to_string(x);
		format.push_back(',');
	}
	format.append(",]");
	return format;
}

onnxutils::DataError onnxutils::check_input(std::vector<int64_t>& input, std::vector<int64_t>& target) {
	if (input.size() != target.size()) {
		return DataError::bad_dims;
	}
	for (int i = 0; i < input.size(); i++) {
		if (target[i] != input[i] && target[i]!=-1) {
			return DataError::bad_shape;
		}
	}
	return DataError::success;
}

/// <param name="input">输入数据的shape数组</param>
/// <param name="target">网络要求数据的shape数组</param>
/// <returns>匹配状态</returns>
onnxutils::DataError onnxutils::check_input(std::vector<std::vector<int64_t>>& input, std::vector<std::vector<int64_t>>& target) {
	if (input.size() != target.size()) {
		return DataError::bad_count;
	}
	for (int i{ 0 }; i < input.size(); i++) {
		auto info = onnxutils::check_input(input[i], target[i]);
		if (info != DataError::success) {
			return info;
		}
		return DataError::success;
	}
}
/// <param name="input">输入数据的shape数组</param>
/// <param name="target">网络要求数据的shape数组</param>
/// <returns>输入描述</returns>
std::string onnxutils::get_error_info(std::vector<std::vector<int64_t>>& input, std::vector<std::vector<int64_t>>& target) {
	std::string info;

	int count = 0;
	for (auto& shape : input) {
		info.append("input: ").append(std::to_string(count)).append(onnxutils::print_shape(shape)).push_back('\n');
		count += 1;
	}
	count = 0;
	for (auto& shape : target) {
		info.append("input target:").append(std::to_string(count)).append(onnxutils::print_shape(shape)).push_back('\n');
		count += 1;
	}
	return info;
}


onnxutils::ModelIOInfo::ModelIOInfo(){
	allocator = nullptr;
	input_count = 0;
	output_count = 0;
}

onnxutils::ModelIOInfo::ModelIOInfo(Ort::Session* session, Ort::AllocatorWithDefaultOptions* allocator_):allocator{ allocator_ }
{
	input_count = session->GetInputCount();
	for (int i = 0; i < input_count; i++) {
		char* input_name = session->GetInputName(i, *allocator);
		char* for_name = new char[strlen(input_name)+1];
		strncpy_s(for_name, sizeof(char) * (strlen(input_name)+1), input_name, sizeof(char) * strlen(input_name));
		allocator->Free(const_cast<void*>(reinterpret_cast<const void*>(input_name)));
		info.append("Input:").append(std::to_string(i)).append(", name = ").append(std::string(for_name));
		input_names.push_back(for_name);

		// 可以得到每个输入的信息
		Ort::TypeInfo type_info = session->GetInputTypeInfo(i);

		// 这个类则是记录了tensor的type和形状信息
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		//enum 类型枚举
		ONNXTensorElementDataType type = tensor_info.GetElementType();
		input_data_format.push_back(type);
		info.append(", type = ").append(std::to_string(type));

		input_shapes.emplace_back(tensor_info.GetShape());
		for (size_t j = 0; j < input_shapes[i].size(); j++)
			info.append(", dim ").append(std::to_string(j)).append(" = ").append(std::to_string(input_shapes[i][j]));
		info.push_back('\n');
	}

	output_count = session->GetOutputCount();
	for (int i = 0; i < output_count; i++) {
		char* output_name = session->GetOutputName(i, *allocator);
		char* for_name = new char[strlen(output_name) + 1];
		strncpy_s(for_name, sizeof(char) * (strlen(output_name) + 1), output_name, sizeof(char) * strlen(output_name));
		allocator->Free(const_cast<void*>(reinterpret_cast<const void*>(output_name)));
		info.append("Output:").append(std::to_string(i)).append(", name = ").append(std::string(for_name));
		output_names.push_back(for_name);

		// 可以得到每个输入的信息
		Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);

		// 这个类则是记录了tensor的type和形状信息
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		//enum 类型枚举
		ONNXTensorElementDataType type = tensor_info.GetElementType();
		output_data_format.push_back(type);
		info.append(", type = ").append(std::to_string(type));

		output_shapes.emplace_back(tensor_info.GetShape());
		for (size_t j = 0; j < output_shapes[i].size(); j++)
			info.append(", dim ").append(std::to_string(j)).append(" = ").append(std::to_string(output_shapes[i][j]));
		info.push_back('\n');
	}
	std::cout << info << std::endl;
}

onnxutils::ModelIOInfo& onnxutils::ModelIOInfo::operator=(onnxutils::ModelIOInfo&& other) {
	if (&other == this) {
		return *this;
	}
	for (int i = 0; i < other.input_names.size(); i++) {
		input_names.push_back(other.input_names[i]);
		other.input_names[i] = nullptr;
	}
	for (int i = 0; i < other.output_names.size(); i++) {
		output_names.push_back(other.output_names[i]);
		other.output_names[i] = nullptr;
	}
	input_count = other.input_count;
	input_data_format = other.input_data_format;
	input_shapes = other.input_shapes;

	output_count = other.output_count;
	output_data_format = other.output_data_format;
	output_shapes = other.output_shapes;
	return *this;
}

onnxutils::ModelIOInfo::~ModelIOInfo() {
#ifdef DEBUG
	std::cout << "析构了ModelInfo" << std::endl;
#endif
	for (char* node_name : input_names)
		delete[] node_name;
	for (char* node_name : output_names)
		delete[] node_name;
}
std::vector<std::vector<int64_t>>& onnxutils::ModelIOInfo::get_input_shapes() {
	return input_shapes;
}
std::vector<char*>& onnxutils::ModelIOInfo::get_input_names() {
	return input_names;
}
std::vector<ONNXTensorElementDataType>& onnxutils::ModelIOInfo::get_input_data_format() {
	return input_data_format;
}
size_t onnxutils::ModelIOInfo::get_input_count() {
	return input_count;
}

std::vector<std::vector<int64_t>>& onnxutils::ModelIOInfo::get_ouput_shapes() {
	return output_shapes;
}
std::vector<char*>& onnxutils::ModelIOInfo::get_out_names() {
	return output_names;
}
std::vector<ONNXTensorElementDataType>& onnxutils::ModelIOInfo::get_output_data_format() {
	return output_data_format;
}
size_t onnxutils::ModelIOInfo::get_output_count() {
	return output_count;
}

const std::string& onnxutils::ModelIOInfo::print_info() {
	return info;
}