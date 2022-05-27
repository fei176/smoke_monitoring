#pragma once
#include "transforms.h"
#include "module.h"
// in classfy, assume model has only "one" image as the inputs 
class CV :
    public Module
{
public:
    CV();
    virtual cv::Mat normalize(const cv::Mat& mat,const std::vector<float>& mean = { 0, 0, 0 }, const std::vector<float>& std = { 1, 1, 1 });
    virtual cv::Mat resize(const cv::Mat&, int w, int h);
    virtual std::vector<Ort::Value> inference(std::vector < Ort::Value >& inputs , Ort::RunOptions&);
};

