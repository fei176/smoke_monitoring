#pragma once

#include "module/resnet.h"
#include "module/resnetxt_ibn.h"
#include "module/densenet.h"
#include "module/inception.h"
#include "module/ghostnet.h"
#include "module/mobilenet.h"
#include "module/sequeezenet.h"
#include "module/shufflenet.h"

#include "module/detection.h"
#include "module/yolo.h"
#include "module/yolox.h"
#include "module/detr.h"
#include "module/yolos.h"
#include "module/ssd.h"
#include "module/yolov6.h"
#include "module/fcos.h"

#include <iostream>
#include <stdlib.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <string>
#include <vector>

#include "http/threadpool.h"


class ModelZoo {
public:
    cv::Mat forward(int id, int h, int w, float nms, float conf, cv::Mat& mat);
    ~ModelZoo();

    static ModelZoo* getInstanse(const char* weights_dir);
    static ModelZoo* getInstanse();

private:
    ModelZoo(const char* weights_dir);
    void check_exist(int id);
    const ORTCHAR_T* get_weight_path(const char*);
    //for model use
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::RunOptions run_option;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions allocator;

    std::unordered_map <int, Detection*> model_record;
    std::unordered_map<int, const ORTCHAR_T*> model_path;

    std::mutex create_mut;
    std::mutex use_mut;

    static ModelZoo* zoo;
    static std::mutex init_mut;
};
