#pragma once
#include "cv.h"
#include "functional/functional.h"
#include "functional/detection.h"

class Yolo :
    public CV
{
public:
    Yolo(int w = 224, int h = 224,
        int max_w = 640, int max_h = 640,
        const std::vector<float>& mean = { 0.485, 0.456, 0.40 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = true, 
        float nms_thresh = 0.5, float confidence_thresh = 0.1,
        int small_thres =2,int big_thres = 7800);

    cv::Mat preprocess(std::string& img_path, transforms::DataFormat data_format);
    std::vector<Ort::Value> forward(std::string& img_path, Ort::MemoryInfo& mem_info, Ort::RunOptions&,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    int postprocess(std::vector<Ort::Value>&);
private:
    int h, w, max_h, max_w;
    std::vector<float> mean, std;
    bool fixed_input;
    std::vector <int64_t> shape;
    // 这个用于解析mat的数据
    std::vector<float> tensor_data;
    // for detect
    float nms_thresh;
    float confidence_thresh;
    int small_thres;
    int big_thres;
};