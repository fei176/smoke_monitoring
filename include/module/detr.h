#pragma once
#include "module/module.h"
#include "functional/functional.h"
#include "functional/detection.h"

class Detr :
    public Module
{
public:
    Detr(int w = 224, int h = 224,
        int max_w = 800, int max_h = 800,
        const std::vector<float>& mean = { 0.485, 0.456, 0.406 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = true, 
        float nms_thresh = 0.5, float confidence_thresh = 0.25);

    std::vector<cv::Mat> preprocess(std::string& img_path, transforms::DataFormat data_format);
    std::vector<Ort::Value> forward(std::string& img_path, Ort::MemoryInfo& mem_info, Ort::RunOptions&,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    std::vector<Box>  postprocess(std::vector<Ort::Value>&);
private:
    int h, w, max_h, max_w;
    std::vector<float> mean, std;
    bool fixed_input;
    std::vector <int64_t> shape;
    std::vector<int64_t> mask_shape;
    // 这个用于解析mat的数据
    std::vector<float> tensor_data;
    // for detect
    float ratio;
    float nms_thresh;
    float confidence_thresh;
};