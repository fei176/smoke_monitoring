#include "functional/functional.h"

void functional::remove_low_confidence(float conf_threshold, std::vector<Box>& boxes,int offset) {
	int slow = 0;
	int fast = 0;
	for (; fast < boxes.size(); fast++) {
		if (boxes[fast].conf > conf_threshold) {
			boxes[slow] = boxes[fast];
			if (offset) {
				boxes[slow].x1 += boxes[slow].label * offset;
				boxes[slow].y1 += boxes[slow].label * offset;
				boxes[slow].x2 += boxes[slow].label * offset;
				boxes[slow].y2 += boxes[slow].label * offset;
			}
			slow += 1;
		}
	}
	boxes.erase(boxes.begin() + slow, boxes.end());
}

// follow the yolo's way to remove the too small or two big box
void functional::remove_low_area(float minimum, float maximum, std::vector<Box>& boxes) {
	int slow = 0;
	int fast = 0;
	for (; fast < boxes.size(); fast++) {
		if ((boxes[fast].w() > minimum && boxes[fast].w() < maximum) && (boxes[fast].h() > minimum && boxes[fast].h() < maximum)) {
			boxes[slow] = boxes[fast];
			slow += 1;
		}
	}
	boxes.erase(boxes.begin() + slow, boxes.end());
}

// an nms impl for Box,same function as template nms 
void functional::nms(std::vector<Box>& boxes, int iou_threshold, int top_k, bool sorted, int offset) {
	if (boxes.empty()) {
		return;
	}
	if (sorted) {
		std::sort(boxes.begin(), boxes.end(), [](Box& box1, Box& box2) {
			return box1.conf > box2.conf;
			});
	}
	int bbox_nums = boxes.size();
	std::vector<bool> removed(bbox_nums, false);
	int slow = 0;
	int fast = 0;
	for (; fast < bbox_nums; fast++) {
		if (removed[fast]) {
			continue;
		}
		for (int j = fast + 1; j < bbox_nums; j++) {
			if (boxes[fast].iou(boxes[j]) > iou_threshold) {
				removed[j] = true;
			}
		}
		boxes[slow] = boxes[fast];
		if (offset) {
			boxes[slow].x1 -= boxes[slow].label * offset;
			boxes[slow].y1 -= boxes[slow].label * offset;
			boxes[slow].x2 -= boxes[slow].label * offset;
			boxes[slow].y2 -= boxes[slow].label * offset;
		}
		slow += 1;
		if (slow > top_k) {
			break;
		}
	}
	boxes.erase(boxes.begin() + slow, boxes.end());
}

// an fuse_nms impl for Box,same function as template nms 
void functional::fuse_nms(std::vector<Box>& boxes, int iou_threshold, int top_k, bool sorted) {
	if (boxes.empty()) {
		return;
	}
	if (sorted) {
		std::sort(boxes.begin(), boxes.end(), [](Box& box1, Box& box2) {
			return box1.conf > box2.conf;
			});
	}
	int bbox_nums = boxes.size();
	std::vector<bool> removed(bbox_nums, false);
	int slow = 0;
	int fast = 0;
	for (; fast < bbox_nums; fast++) {
		if (removed[fast]) {
			continue;
		}
		std::vector<Box> remove_bbox{ boxes[fast] };
		float total = std::exp(boxes[fast].conf);
		for (int j = fast + 1; j < bbox_nums; j++) {
			if (boxes[fast].iou(boxes[j]) > iou_threshold) {
				removed[j] = true;
				remove_bbox.push_back(boxes[j]);
				total += std::exp(boxes[j].conf);
			}
		}
		Box fused_box;
		for (auto& box : remove_bbox) {
			float rate = std::exp(box.conf) / total;
			fused_box.x1 += box.x1 * rate;
			fused_box.x2 += box.x2 * rate;
			fused_box.y1 += box.y1 * rate;
			fused_box.y2 += box.y2 * rate;
		}
		boxes[slow] = fused_box;
		slow += 1;
		if (slow > top_k) {
			break;
		}
	}
	boxes.erase(boxes.begin() + slow, boxes.end());
}

void functional::draw_box(cv::Mat& img, std::vector<Box>& boxes,int mode) {
	if (boxes.empty()) {
		return;
	}
	for (auto& box : boxes) {
		cv::rectangle(img, box.rect(), DetectionHelper::get_color(box.label),2);
		std::string text = DetectionHelper::get_name_coco(box.label, mode);
		text += std::to_string(box.conf).substr(0, 4);
		cv::putText(img, text, box.text(), cv::FONT_HERSHEY_SIMPLEX, 0.6f, DetectionHelper::get_color(box.label));
	}
}
