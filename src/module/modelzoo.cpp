#include "module/modelzoo.h"

using namespace Ort;

ModelZoo* ModelZoo:: zoo = nullptr;
std::mutex ModelZoo::init_mut;

ModelZoo* ModelZoo::getInstanse(const char* weights_dir) {
	if (zoo == nullptr) {
		std::unique_lock<std::mutex> init(init_mut);
		if (zoo == nullptr) {
			zoo = new ModelZoo(weights_dir);
		}
	}
	return zoo;
}

ModelZoo* ModelZoo::getInstanse() {
	return zoo;
}

ModelZoo::~ModelZoo() {
	for (auto c : model_path) {
		free((void*)c.second);
	}
	for (auto& x : model_record) {
		delete x.second;
	}
}

const ORTCHAR_T* ModelZoo::get_weight_path(const char* path) {
#ifdef _WIN32
	size_t size = strlen(path) + 1;
	wchar_t* w = (wchar_t*)malloc(size * sizeof(wchar_t));
	mbstowcs(w, path, size);
	return w;
#elif __APPLE__ || __linux__
	char* tmp = (char*)(malloc(sizeof(char) * strlen(path) + 1));
	memcpy(tmp,path,strlen(path));
	return tmp;
#endif
}

ModelZoo::ModelZoo(const char* weights_dir)
	:memory_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) },
	env{ ORT_LOGGING_LEVEL_WARNING, "test" }{
	// 以后可以写个根据配置文件来加载的机制和动态加载，目前只加载固定目录下的文件
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	model_path.insert(std::make_pair(0, get_weight_path((std::string(weights_dir) + "/detection/yolov5.onnx").c_str())));
	model_path.insert(std::make_pair(1, get_weight_path((std::string(weights_dir) + "/detection/yolox.onnx").c_str())));
	model_path.insert(std::make_pair(2, get_weight_path((std::string(weights_dir) + "/detection/detr.onnx").c_str())));
	model_path.insert(std::make_pair(3, get_weight_path((std::string(weights_dir) + "/detection/yolos.onnx").c_str())));
	model_path.insert(std::make_pair(4, get_weight_path((std::string(weights_dir) + "/detection/ssd.onnx").c_str())));
	model_path.insert(std::make_pair(5, get_weight_path((std::string(weights_dir) + "/detection/yolov6.onnx").c_str())));
	model_path.insert(std::make_pair(6, get_weight_path((std::string(weights_dir) + "/detection/fcos.onnx").c_str())));
	std::cout << "model zoo init successed " << std::endl;
};

void ModelZoo::check_exist(int id) {
	std::unique_lock<std::mutex> lock(create_mut);
	auto res = model_record.find(id);
	if (res == model_record.end()) {
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
		else if (id == 3) {
			YoloS* t = new YoloS();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else if (id == 4) {
			SSD* t = new SSD();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else if (id == 5) {
			Yolo6* t = new Yolo6();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
		else if (id == 6) {
			FCOS* t = new FCOS();
			t->Init(env, session_options, model_path[id], allocator);
			model_record[id] = t;
		}
	}
}

cv::Mat ModelZoo::forward(int id, int h, int w, float nms, float conf, cv::Mat& mat){
	check_exist(id);
	std::unique_lock<std::mutex> lock(use_mut);
	Detection* model = model_record[id];
	model->adjust_par(h, w, nms, conf);
	return model->call(mat, memory_info, run_option);
}