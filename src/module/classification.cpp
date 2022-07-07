#include <module/classification.h>

Classify::Classify(int w, int h,
    int max_w, int max_h,
    const std::vector<float>& mean,
    const std::vector<float>& std,
    bool fixed_input, bool batch) {
    this->w = w;
    this->h = h;
    if (w > max_w)
        this->max_w = (int(w / 32) + 1) * 32;
    else
        this->max_w = max_w;
    if (h > max_h)
        this->max_h = (int(h / 32) + 1) * 32;
    else
        this->max_h = max_h;
    this->mean = mean;
    this->std = std;
    tensor_data = std::move(std::vector<float>(max_w * max_h * 3, 0));
    shape = { 1,3,h,w };

    this->fixed_input = fixed_input;
    // batch is not support yet.
    this->batch = batch;
}

cv::Mat Classify::preprocess(std::string& img_path, transforms::DataFormat data_format) {
    cv::Mat mat = cv::imread(img_path);
    if (!onnxutils::check_data(mat, model_info.get_input_shapes()[0], data_format)) {
        throw std::runtime_error("input shape batch or channel don't match the target input");
    }
    mat.convertTo(mat, CV_32FC3);
    mat /= 255;
    cv::Mat resize_img = transforms::resize(mat, fixed_input, w, h);
    return transforms::normalize(resize_img, mean, std);
}

// in classification, assume input is a picture,it's output ia a vector.
// we transform it to float, than to 0-1, than resize with or without padding, finally normalize it.
// preprocess -> inference -> postprocess -> return
int Classify::forward(std::string& img_path, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_info,
    transforms::DataFormat data_format) {
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(transforms::make_tensor(preprocess(img_path, data_format),
        shape, mem_info, tensor_data, transforms::DataFormat::CHW));

    std::vector<Ort::Value> outputs = inference(inputs, run_info);

    return postprocess(outputs);
}
int Classify::postprocess(std::vector<Ort::Value>& result) {
    Ort::Value& res_value = result[0];
    size_t out_size = res_value.GetTensorTypeAndShapeInfo().GetElementCount();
    float* res = res_value.GetTensorMutableData<float>();
    return functional::argmax<float>(res, out_size);
}