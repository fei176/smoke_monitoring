#include "module/module.h"

using namespace std;

Module::Module(){
	
}

Module::~Module() {
	
}

void Module::Init(Ort::Env& env, Ort::SessionOptions& session_op, const ORTCHAR_T* model_path,Ort::AllocatorWithDefaultOptions& allocator) {
	session = make_shared<Ort::Session>(env, model_path, session_op);
	model_info = std::move(onnxutils::ModelIOInfo(&(*session), &allocator));
}

std::vector<Ort::Value> Module::inference(std::vector < Ort::Value >& inputs, Ort::RunOptions& run_option){
	return session->Run(run_option, model_info.get_input_names().data(), inputs.data(),
		model_info.get_input_count(), model_info.get_out_names().data(), model_info.get_output_count());
}