#pragma once
#include "cv.h"
#include "functional/functional.h"

class Resnet :
    public CV
{
public:
    Resnet(int w = 224, int h = 224,
        int max_w=640,int max_h=640,
        const std::vector<float>& mean = { 0.485, 0.456, 0.40 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = true);

    cv::Mat preprocess(std::string& img_path, transforms::DataFormat data_format);
    std::vector<Ort::Value> forward(std::string& img_path,Ort::MemoryInfo & mem_info, Ort::RunOptions&, 
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    int postprocess(std::vector<Ort::Value>&);
private:
    int h, w,max_h,max_w;
    std::vector<float> mean, std;
    bool fixed_input;
    std::vector <int64_t> shape;
    // 这个用于解析mat的数据
    std::vector<float> tensor_data;
};

