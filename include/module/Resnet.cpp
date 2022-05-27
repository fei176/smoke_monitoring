#include "module/Resnet.h"

Resnet::Resnet(int w , int h,
    int max_w, int max_h,
    const std::vector<float>& mean ,
    const std::vector<float>& std ,
    bool fixed_input) {
    this->w = w;
    this->h = h;
    this->max_h = max_h;
    this->max_w = max_w;
    this->mean = mean;
    this->std = std;
    this->fixed_input = fixed_input;
    tensor_data = std::move(std::vector<float>(max_w * max_h * 3, 0));
    shape = { 1,3,h,w };
}


cv::Mat Resnet::preprocess(std::string& img_path, transforms::DataFormat data_format) {
    cv::Mat mat = cv::imread(img_path);
    //输入形状检查
    if (!transforms::check_data(mat, model_info.get_input_shapes()[0], data_format)) {
        throw std::runtime_error("input shape batch or channel don't match the target input");
    }
    mat.convertTo(mat, CV_32FC3);
    mat /= 255;

    cv::Mat resize_img = transforms::resize(mat, false, w);
    // 构造数据
    // 1 强制缩放到固定比例（w=h)
    // 2 缩放到32倍数的最近大小,min(32x,640),补零
    // normalize
    return normalize(resize_img, mean, std);
}

std::vector<Ort::Value> Resnet::forward(std::string& img_path, Ort::MemoryInfo& mem_info,Ort::RunOptions & run_info,
    transforms::DataFormat data_format) {
    std::vector<Ort::Value> inputs;
    
    inputs.emplace_back(transforms::to_tensor(preprocess(img_path,data_format), 
        shape, mem_info, tensor_data, transforms::DataFormat::CHW));

    return inference(inputs, run_info);
    return inputs;
}

int Resnet::postprocess(std::vector<Ort::Value>& result) {
    Ort::Value& res_value = result[0];
    float* res = res_value.GetTensorMutableData<float>();
    auto& output_shape = model_info.get_ouput_shapes();
    std::vector<int64_t>& shape = output_shape[0];
    int classes = shape[shape.size() - 1];
    return functional::argmax<float>(res, classes);
}
