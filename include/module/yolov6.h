#pragma once
#include "module/detection.h"

class Yolo6 :
    public Detection
{
public:
    Yolo6(int w = 640, int h = 640,
        int max_w = 1280, int max_h = 1280,
        const std::vector<float>& mean = { 0., 0., 0. },
        const std::vector<float>& std = { 1., 1., 1. },
        float nms_thresh = 0.5, float confidence_thresh = 0.25);
    cv::Mat call(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    std::vector<Box> call_box(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_options,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    void adjust_par(int h = 640, int w = 640, float nms_thresh = 0.45, float confidence_thresh = 0.2);

private:
    cv::Mat preprocess(cv::Mat& img, transforms::DataFormat data_format);
    std::vector<Ort::Value> forward(cv::Mat& img, Ort::MemoryInfo& mem_info, Ort::RunOptions& run_info,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    std::vector<Box> postprocess(std::vector<Ort::Value>&);

    int h, w, max_h, max_w;
    std::vector<float> mean, std;
    std::vector <int64_t> shape;
    // 这个用于解析mat的数据
    std::vector<float> tensor_data;
    // for detect
    int u_padding;
    int l_padding;
    float ratio;
    float nms_thresh;
    float confidence_thresh;
};