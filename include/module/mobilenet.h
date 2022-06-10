#pragma once
#include "module/classification.h"

class Mobilenet :
    public Classify
{
public:
    Mobilenet(int w = 224, int h = 224,
        int max_w=640,int max_h=640,
        const std::vector<float>& mean = { 0.485, 0.456, 0.40 },
        const std::vector<float>& std = { 0.229, 0.224, 0.225 },
        bool fixed_input = false,bool batch=false);
};

