#include "module/modelzoo.h"

using namespace Ort;

ModelZoo* ModelZoo::getInstanse() {
	static ModelZoo modelzoo;
	return &modelzoo;
}

ModelZoo::~ModelZoo() {
	Yolo* t = (Yolo*)model_record[0];
	delete t;
	YoloX* t1 = (YoloX*)model_record[0];
	delete t1;
	Detr* t2 = (Detr*)model_record[0];
	delete t2;
}

ModelZoo::ModelZoo() 
	:memory_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) },
	env{ ORT_LOGGING_LEVEL_WARNING, "test" }{

    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	model_path.insert(std::make_pair(0, L"C:/project/detection/yolov5.onnx"));
	model_path.insert(std::make_pair(1, L"C:/project/detection/yolox.onnx"));
	model_path.insert(std::make_pair(2, L"C:/project/detection/detr.onnx"));

	model_path.insert(std::make_pair(20, L"C:/project/classify/densenet.onnx"));
	model_path.insert(std::make_pair(21, L"C:/project/detection/ghost.onnx"));
	model_path.insert(std::make_pair(22, L"C:/project/detection/inception.onnx"));
	model_path.insert(std::make_pair(23, L"C:/project/detection/mobilenet.onnx"));
	model_path.insert(std::make_pair(24, L"C:/project/detection/resnet.onnx"));
	model_path.insert(std::make_pair(25, L"C:/project/detection/resnext101_ibn.onnx"));
	model_path.insert(std::make_pair(26, L"C:/project/detection/sequeezenet.onnx"));
	model_path.insert(std::make_pair(27, L"C:/project/detection/shufflenet.onnx"));
};


cv::Mat ModelZoo::forward(int id, int h, int w, float nms, float conf, cv::Mat& mat){
	auto res = model_record.find(id);
	if (res == model_record.end()) {

		void* model = nullptr;
		if (id == 0) {
			Yolo* t = new Yolo();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else if (id == 1) {
			YoloX* t = new YoloX();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else if (id == 2) {
			Detr* t = new Detr();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else {
		}
	}

	void* model = model_record[id];
	if (id == 0) {
		std::unique_lock<std::mutex> lock(yolov5_mut);
		Yolo* t = (Yolo*)model;
		t->adjust_par(h, w, nms, conf);
		return t->call(mat,memory_info, run_option);
	}
	else if (id == 1) {
		std::unique_lock<std::mutex> lock(yolo_mut);
		YoloX* t = (YoloX*)model;
		t->adjust_par(h, w, nms, conf);
		return t->call(mat, memory_info, run_option);
	}
	else if (id == 2) {
		std::unique_lock<std::mutex> lock(detr_mut);
		Detr* t = (Detr*)model;
		t->adjust_par(h, w, nms, conf);
		return t->call(mat, memory_info, run_option);
	}
}