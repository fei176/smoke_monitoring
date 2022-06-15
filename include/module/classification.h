#pragma once
#include "functional/transforms.h"
#include "functional/functional.h"
#include "module.h"

// in classfy, assume model has only "one" image as the inputs 
class Classify :
    public Module
{
public:
    Classify(int w = 224, int h = 224,
        int max_w = 640, int max_h = 640,
        const std::vector<float>& mean = { 0.485, 0.456, 0.40 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = false,bool batch = false);

    virtual cv::Mat preprocess(std::string& img_path, transforms::DataFormat data_format);
    int forward(std::string& img_path, Ort::MemoryInfo& mem_info, Ort::RunOptions&,
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    virtual int postprocess(std::vector<Ort::Value>&);
private:
    int h, w, max_h, max_w;
    std::vector<float> mean, std;
    bool fixed_input;
    bool batch;
    std::vector <int64_t> shape;
    // 这个用于解析mat的数据
    std::vector<float> tensor_data;
};

