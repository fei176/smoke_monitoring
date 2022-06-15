#include "module/module.h"

using namespace std;

Module::Module(){
	
}

Module::~Module() {
	
}

// 这里可以考虑一下是否需要用异步，因为必须加载完成才能使用，在UI中，需要在另一个线程中使用，否则会卡住主线程。
// 是否需要回调，如何包装函数和参数呢。
void Module::Init(Ort::Env& env, Ort::SessionOptions& session_op, const wchar_t* model_path,Ort::AllocatorWithDefaultOptions& allocator) {
	session = make_shared<Ort::Session>(env, model_path, session_op);
	model_info = std::move(onnxutils::ModelIOInfo(&(*session), &allocator));
}

std::vector<Ort::Value> Module::inference(std::vector < Ort::Value >& inputs, Ort::RunOptions& run_option){
	return session->Run(run_option, model_info.get_input_names().data(), inputs.data(),
		model_info.get_input_count(), model_info.get_out_names().data(), model_info.get_output_count());
}