#include "module/detr.h"

Detr::Detr(int w, int h,
    int max_w, int max_h,
    const std::vector<float>& mean,
    const std::vector<float>& std,
    float nms_thresh, float confidence_thresh) {
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
    this->nms_thresh = nms_thresh;
    this->confidence_thresh = confidence_thresh;
}

cv::Mat Detr::call(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
    transforms::DataFormat data_format) {
    std::vector<Box> pred_classes = call_box(img, mem_info, run_options, data_format);
    img *= 255;
    img.convertTo(img, CV_8UC3);
    functional::draw_box(img, pred_classes, 80);
    return img;
}

std::vector<Box> Detr::call_box(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
    transforms::DataFormat data_format) {
    auto result = forward(img, mem_info, run_options, data_format);
    return postprocess(result);
}

void Detr::adjust_par(int h, int w, float nms_thresh, float confidence_thresh) {
    this->h = h;
    this->w = w;
    this->nms_thresh = nms_thresh;
    this->confidence_thresh = confidence_thresh;
}

std::vector<cv::Mat> Detr::preprocess(cv::Mat& mat, transforms::DataFormat data_format) {
    //������״���,�쳣�ڴ��׳�
    if (!onnxutils::check_data(mat, model_info.get_input_shapes()[0], data_format)) {
        cv::Mat adjust;
        if (!onnxutils::tyr_adjust_channel(mat, adjust, model_info.get_input_shapes()[0][1])) {
            throw std::runtime_error("input shape batch or channel don't match the target input");
        }
        mat = adjust;
    }
    mat.convertTo(mat, CV_32FC3);
    mat /= 255;
    cv::Size new_size = transforms::get_new_size(mat, std::max(h,w),&ratio);
    cv::Mat resize_img;
    shape = { 1,3,new_size.height,new_size.width };
    mask_shape = { 1,new_size.height,new_size.width };
    cv::resize(mat, resize_img, new_size);
    cv::Mat norm_img;
    transforms::normalize(resize_img, norm_img,mean.data(), std.data());
    cv::Mat mask(new_size,CV_32FC1,cv::Scalar(0.,0.,0.));
    return { resize_img,mask };
}

std::vector<Ort::Value> Detr::forward(cv::Mat& one_img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_info,
    transforms::DataFormat data_format) {
    std::vector<cv::Mat> img = preprocess(one_img, data_format);
    return forward(img, mem_info, run_info);
}

std::vector<Ort::Value> Detr::forward(std::vector<cv::Mat>& input_cv, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_info,
    transforms::DataFormat data_format) {
    std::vector<Ort::Value> inputs;
    inputs.push_back(transforms::make_tensor(input_cv[0], shape, mem_info, tensor_data, transforms::DataFormat::CHW));
    inputs.push_back(transforms::make_tensor(input_cv[1], mask_shape, mem_info, tensor_data, transforms::DataFormat::HWC));
    return inference(inputs, run_info);
}

// yolo has one output
std::vector<Box> Detr::postprocess(std::vector<Ort::Value>& result) {
    float* pre_result = result[0].GetTensorMutableData<float>();
    //��value�ϻ�ȡʵ�ʵĴ�С����Ԫ��������
    size_t out_size = result[0].GetTensorTypeAndShapeInfo().GetElementCount();

    int box_size = out_size / 6;
    std::vector<Box> bboxs(box_size);
    memcpy(bboxs.data(), pre_result, sizeof(float) * out_size);
    functional::remove_low_confidence(confidence_thresh, bboxs);
    functional::nms(bboxs, nms_thresh, 30, true);
    for (auto& box : bboxs) {
        box.label -= 1;
        box.resize2img(ratio, 0, 0);
        box.resize2ori(shape[2], shape[3]);
    }
    return bboxs;
}
