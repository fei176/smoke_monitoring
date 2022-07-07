#include "core/onnxutils.h"


std::string onnxutils::print_shape(std::vector<int64_t>& shape) {
	std::string format="shape :[";
	for (auto x : shape) {
		format += std::to_string(x);
		format.push_back(',');
	}
	format.append(",]");
	return format;
}

bool onnxutils::check_data(const std::vector<int64_t> input, const std::vector<int64_t>& target) {
	return onnxutils::check_input(input, target) == onnxutils::DataError::success;
}

bool onnxutils::check_data(const cv::Mat& image, const std::vector<int64_t>& shape,
	const transforms::DataFormat& data_format) {
	unsigned int h = image.rows;
	unsigned int w = image.cols;
	unsigned int c = image.channels();

	std::vector<int64_t> input_shape;
	if (data_format == transforms::DataFormat::CHW) {
		input_shape = { 1, c,h,w };
	}
	else {
		input_shape = { 1,h,w,c };
	}
	return onnxutils::check_input(input_shape, shape) == onnxutils::DataError::success;
}

onnxutils::DataError onnxutils::check_input(const std::vector<int64_t>& input, const std::vector<int64_t>& target) {
	if (input.size() != target.size()) {
		return onnxutils::DataError::bad_dims;
	}
	for (int i = 0; i < input.size(); i++) {
		if (target[i] != input[i] && target[i]!=-1) {
			return DataError::bad_shape;
		}
	}
	return DataError::success;
}

onnxutils::DataError onnxutils::check_input(const std::vector<std::vector<int64_t>>& input, const std::vector<std::vector<int64_t>>& target) {
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

bool onnxutils::tyr_adjust_channel(cv::Mat& ori, cv::Mat& out, int channels) {
	unsigned int c = ori.channels();
	if (c == 1 && channels == 3) {
		cv::cvtColor(ori, out, cv::COLOR_GRAY2RGB);
	}
	else if (c == 4 && channels == 3) {
		cv::cvtColor(ori, out, cv::COLOR_BGRA2RGB);
	}
	else {
		return false;
	}
	return true;
}

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
	print_flag=true;
}

onnxutils::ModelIOInfo::ModelIOInfo(Ort::Session* session, Ort::AllocatorWithDefaultOptions* allocator_,bool print_flag):allocator{ allocator_ }
{	
	input_count = session->GetInputCount();
	for (int i = 0; i < input_count; i++) {
		char* input_name = session->GetInputName(i, *allocator);
		char* for_name = new char[strlen(input_name)+1];
		for_name[strlen(input_name)] = '\0';
		memcpy(for_name,input_name,sizeof(char) * strlen(input_name));
		allocator->Free(const_cast<void*>(reinterpret_cast<const void*>(input_name)));
		info.append("Input:").append(std::to_string(i)).append(", name = ").append(std::string(for_name));
		input_names.push_back(for_name);

		Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
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
		memcpy(for_name,output_name,sizeof(char) * strlen(output_name));
		for_name[strlen(output_name)] = '\0';
		allocator->Free(const_cast<void*>(reinterpret_cast<const void*>(output_name)));
		info.append("Output:").append(std::to_string(i)).append(", name = ").append(std::string(for_name));
		output_names.push_back(for_name);

		Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);

		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		output_data_format.push_back(type);
		info.append(", type = ").append(std::to_string(type));

		output_shapes.emplace_back(tensor_info.GetShape());
		for (size_t j = 0; j < output_shapes[i].size(); j++)
			info.append(", dim ").append(std::to_string(j)).append(" = ").append(std::to_string(output_shapes[i][j]));
		info.push_back('\n');
	}
	if(print_flag){
		std::cout << info << std::endl;
	}
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
	std::cout << "ModelInfo" << std::endl;
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