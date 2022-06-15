#pragma once

#include "module/resnet.h"
#include "module/resnetxt_ibn.h"
#include "module/densenet.h"
#include "module/inception.h"
#include "module/ghostnet.h"
#include "module/mobilenet.h"
#include "module/sequeezenet.h"
#include "module/shufflenet.h"

#include "module/yolo.h"
#include "module/yolox.h"
#include "module/detr.h"

#include <iostream>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <string>
#include <vector>

#include "http/threadpool.h"


class ModelZoo {
public:
    static ModelZoo* getInstanse();
    cv::Mat forward(int id, int h, int w, float nms, float conf, cv::Mat& mat);
    ~ModelZoo();
private:
    ModelZoo();
    //for model use
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::RunOptions run_option;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions allocator;

    std::unordered_map <int, void*> model_record;
    std::unordered_map<int, const wchar_t*> model_path;

    std::mutex yolo_mut;
    std::mutex yolov5_mut;
    std::mutex detr_mut;
};

