#include "module/ssd.h"

SSD::SSD(float nms_thresh, float confidence_thresh) {
    this->fix_w = 512;
    this->fix_h = 512;
    this->mean = { 123.0, 117.0, 104.0 };
    tensor_data = std::move(std::vector<float>(fix_w * fix_h * 3, 0));
    shape = { 1,3,fix_h,fix_w };

    this->nms_thresh = nms_thresh;
    this->confidence_thresh = confidence_thresh;
}

cv::Mat SSD::call(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
    transforms::DataFormat data_format) {
    std::vector<Box> pred_classes = call_box(img, mem_info, run_options, data_format);
    std::vector<float> tmp_mean{ mean };
    for (auto& x : tmp_mean) {
        x = -x;
    }
    transforms::normalize(img, tmp_mean);
    img.convertTo(img, CV_8UC3);
    functional::draw_box(img, pred_classes, 80);
    return img;
}

std::vector<Box> SSD::call_box(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
    transforms::DataFormat data_format) {
    auto result = forward(img, mem_info, run_options, data_format);
    return postprocess(result);
}

void SSD::adjust_par(int h, int w, float nms_thresh, float confidence_thresh) {
    this->nms_thresh = nms_thresh;
    this->confidence_thresh = confidence_thresh;
}

cv::Mat SSD::preprocess(cv::Mat& mat, transforms::DataFormat data_format) {
    if (!onnxutils::check_data(mat, model_info.get_input_shapes()[0], data_format)) {
        cv::Mat adjust;
        if (!onnxutils::tyr_adjust_channel(mat, adjust, model_info.get_input_shapes()[0][1])) {
            throw std::runtime_error("input shape batch or channel don't match the target input");
        }
        mat = adjust;
    }
    mat.convertTo(mat, CV_32FC3);
    transforms::normalize(mat, mean);
    cv::Mat resize_img = transforms::resize(mat, false, fix_w, fix_h, &u_padding, &l_padding, &ratio, false, cv::Scalar(mean[0], mean[1], mean[2]));
    return resize_img;
}

std::vector<Ort::Value> SSD::forward(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_info,
    transforms::DataFormat data_format) {
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(transforms::make_tensor(preprocess(img, data_format),
        shape, mem_info, tensor_data, transforms::DataFormat::CHW));
    return inference(inputs, run_info);
}

// yolo has one output
std::vector<Box> SSD::postprocess(std::vector<Ort::Value>& result) {
    float* pre_result = result[0].GetTensorMutableData<float>();
    //在value上获取实际的大小，总元素数量。
    size_t out_size = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
    int box_size = out_size / 6;
    std::vector<Box> bboxs(box_size);
    memcpy(bboxs.data(), pre_result, sizeof(float) * out_size);
    functional::remove_low_confidence(confidence_thresh, bboxs);
    functional::nms(bboxs, nms_thresh, 100, true);
    for (auto& box : bboxs) {
        box.resize2ori(shape[2], shape[3]);
        box.resize2img(ratio, u_padding, l_padding);
    } 
    return bboxs;
}
