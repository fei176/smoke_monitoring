#include "CV.h"

CV::CV() {

}

cv::Mat CV::normalize(const cv::Mat& mat, const std::vector<float> &mean, const std::vector<float>& std) {
    cv::Mat image;
    if ((mean.size() != 1 || std.size()!=1) && (mean.size() != mat.channels() || mean.size()!=std.size())) {
        return image;
    }
    std::vector<float> mean_ = mean;
    std::vector<float> std_ = std;
    if (mean.size() == 1) {
        for (int i = 1; i < mat.channels(); i++) {
            mean_.push_back(mean[0]);
            std_.push_back(mean[0]);
        }
    }
    transforms::normalize(mat, image, mean_.data(), std_.data());
    return image;
}

cv::Mat CV::resize(const cv::Mat& mat, int w=224, int h=224) {
    cv::Mat resize_image;
    cv::resize(mat, resize_image, cv::Size(w, h));
    return resize_image;
}

//这里可以考虑用多异步或者多线程
std::vector<Ort::Value> CV::inference(std::vector < Ort::Value >& inputs, Ort::RunOptions& run_option) {
    return session->Run(run_option, model_info.get_input_names().data(), inputs.data(),
        model_info.get_input_count(), model_info.get_out_names().data(), model_info.get_output_count());
}

//batch image 的操作...