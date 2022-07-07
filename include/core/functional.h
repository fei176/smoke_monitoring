#pragma once
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "module/detection_util.h"

namespace functional {
	// classify
	template<typename T>
	int argmax(const std::vector<T>& data) {
		T& max = data.front();
		auto index = data.begin();
		for (auto begin = data.begin(); begin != data.end(); begin++) {
			if (*begin > max) {
				index = begin;
				max = *begin;
			}
		}
		return int(index - data.begin());
	}

	template<typename T>
	int argmax(const T* data, int size) {
		T max = *data;
		int index = 0;
		for (int i = 0; i < size; i++) {
			if (data[i] > max) {
				max = data[i];
				index = i;
			}
		}
		return index;
	}

	// detection
	template<typename T>
	T iou(std::vector<T>& bbox1, std::vector<T>& bbox2) {
		T area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		T area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		T x1 = std::max(bbox1[0], bbox2[0]);
		T y1 = std::max(bbox1[1], bbox2[1]);
		T x2 = std::min(bbox1[2], bbox2[2]);
		T y2 = std::min(bbox1[3], bbox2[3]);
		T area_union = std::max((y2 - y1),(T)0) * std::max((x2 - x1),(T)0);
		return area_union * 1.0 / (area1 + area2 - area_union);
	}

	void remove_low_confidence(float conf_threshold, std::vector<Box>& boxes, int offset = 640);

	// follow the yolo's way to remove the too small or two big box
	void remove_low_area(float minimum, float maximum, std::vector<Box>& boxes);

	// an nms impl for Box,same function as template nms 
	void nms(std::vector<Box>& boxes, int iou_threshold, int top_k, bool sorted, int offset = 640);

	// an fuse_nms impl for Box,same function as template nms 
	void fuse_nms(std::vector<Box>& boxes, int iou_threshold, int top_k, bool sorted);

 	// 尽管类别一般是int,为了方法，在用vector<T>进行处理时，全部选择了float，格式为x1 y1 x2 y2 conf classes,或者可以使用Box类描述框。
	// 这样的nms只能处理单个类别，或者像yolo中一样，手动把多个类之间的距离拉开才行。
	template<typename T>
	void nms(std::vector < std::vector<T> >& input, std::vector<std::vector<T>> &output,int iou_threshold,int top_k,bool sorted) {
		if (input.empty())
			return;
		if (sorted) {
			std::sort(input.begin(), input.end(), [](std::vector<T>& a, std::vector<T>& b) {return a[4] > b[4]; });
		}
		int bbox_nums = input.size();
		std::vector<bool> removed(bbox_nums, false);
		for (int i = 0; i < bbox_nums; i++) {
			if (removed[i])
				continue;
			for (int j = i + 1; j < bbox_nums; j++) {
				if (iou(input[i], input[j]) * 1.0 > iou_threshold) {
					removed[j] = true;
				}
			}
			output.push_back(input[i]);
			if (output.size() > top_k)
				break;
		}
	}

	template<typename T>
	void fuse_nms(std::vector < std::vector<T> >& input, std::vector<std::vector<T>>& output, int iou_threshold, int top_k, bool sorted) {
		if (input.empty())
			return;
		if (sorted) {
			std::sort(input.begin(), input.end(), [](std::vector<T>& a, std::vector<T>& b) {return a[4] > b[4]; });
		}
		int bbox_nums = input.size();
		std::vector<bool> removed(bbox_nums, false);
		for (int i = 0; i < bbox_nums; i++) {
			if (removed[i])
				continue;
			T total = std::exp(input[i][4]);
			std::vector<std::vector<T>> remove_bbox{input[i]};
			for (int j = i + 1; j < bbox_nums; j++) {
				if (iou(input[i], input[j]) * 1.0 > iou_threshold) {
					removed[j] = true;
					remove_bbox.push_back(input[i]);
					total += std::exp(remove_bbox[4]);
				}
			}
			std::vector<T> fuse_bbox = { 0,0,0,0,0,input[i][5]};
			for (auto& bbox : remove_bbox) {
				T rate = std::exp(bbox[4]) / total;
				fuse_bbox[0] += bbox[0] * rate;
				fuse_bbox[1] += bbox[1] * rate;
				fuse_bbox[2] += bbox[2] * rate;
				fuse_bbox[3] += bbox[3] * rate;
			}
			output.emplace_back(fuse_bbox);
			if (output.size() > top_k)
				break;
		}
	}

	void draw_box(cv::Mat& img, std::vector<Box>& boxes, int mode);
}