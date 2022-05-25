#include "Resnet.h"

Resnet::Resnet(int w , int h,
    const std::vector<float>& mean ,
    const std::vector<float>& std ,
    bool fixed_input) {
    this->w = w;
    this->h = h;
    this->mean = mean;
    this->std = std;
    this->fixed_input = fixed_input;
    tensor_data = std::move(std::vector<float>(w * h * 3, 0));
}


std::vector<Ort::Value> Resnet::forward(std::string& img_path, Ort::MemoryInfo& mem_info,Ort::RunOptions & run_info,
    transforms::DataFormat data_format) {
    cv::Mat mat = cv::imread(img_path);
    mat.convertTo(mat, CV_32FC3);
    mat /= 255;
    std::vector<Ort::Value> inputs;
    cv::Mat img;
    if (mat.data) {
        cv::Mat resize_img;
        unsigned int i_h = mat.rows;
        unsigned int i_w = mat.cols;
        unsigned int i_c = mat.channels();
        std::vector<int64_t> input_shapes;
        if (data_format == transforms::DataFormat::CHW) {
            input_shapes.insert(input_shapes.begin(), { i_c,i_h,i_w });
        }
        else {
            input_shapes.insert(input_shapes.begin(), { i_h,i_w,i_c });
        }
        if (onnxutils::check_input(input_shapes, model_info.get_input_shapes()[0]) != onnxutils::DataError::success) {
            // this mean must resize img to w h or model dont support dynamic inputs
            img = normalize(resize(mat, w, h), mean, std);
        }
        else {
            int need_size = i_h * i_w * i_c;
            if (need_size > tensor_data.size()) {
                tensor_data.resize(need_size);
            }
            img = normalize(mat, mean, std);
        }
        inputs.emplace_back(transforms::to_tensor(img, model_info.get_input_shapes()[0], mem_info, tensor_data, transforms::DataFormat::CHW));
        return inference(inputs, run_info);
    }
    std::vector<Ort::Value> a;
    return a;
}

int Resnet::postprocess(std::vector<Ort::Value>& result) {
    Ort::Value& res_value = result[0];
    float* res = res_value.GetTensorMutableData<float>();
    auto& output_shape = model_info.get_ouput_shapes();
    std::vector<int64_t>& shape = output_shape[0];
    int classes = shape[shape.size() - 1];
    return functional::argmax<float>(res, classes);
}
