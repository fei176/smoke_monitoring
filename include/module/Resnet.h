#pragma once
#include "CV.h"
#include "functional.h"

class Resnet :
    public CV
{
public:
    Resnet(int w = 224, int h = 224,
        const std::vector<float>& mean = { 0.485, 0.456, 0.40 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = true);

    
    std::vector<Ort::Value> forward(std::string& img_path,Ort::MemoryInfo & mem_info, Ort::RunOptions&, 
        transforms::DataFormat data_format = transforms::DataFormat::CHW);
    int postprocess(std::vector<Ort::Value>&);
private:
    int h, w;
    std::vector<float> mean, std;
    bool fixed_input;
    // 这个由于解析mat的数据
    std::vector<float> tensor_data;
};

