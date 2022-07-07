#include "module/ghostnet.h"

Ghostnet::Ghostnet(int w, int h,
    int max_w, int max_h,
    const std::vector<float>& mean,
    const std::vector<float>& std,
    bool fixed_input, bool batch) :Classify(w, h, max_w, max_h, mean, std, fixed_input, batch) {
}
