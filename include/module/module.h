#pragma once
#include <string>
#include <stdio.h>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "core/onnxutils.h"

class Module
{
public:
	Module();
	Module(Module& m) = delete;
	Module& operator=(Module& m) = delete;
	virtual ~Module();
	
	void Init(Ort::Env&,Ort::SessionOptions&,const ORTCHAR_T* model_path, Ort::AllocatorWithDefaultOptions& allocator);
	std::vector<Ort::Value> inference(std::vector < Ort::Value >& inputs, Ort::RunOptions& run_option);
protected:
	onnxutils::ModelIOInfo model_info;
	std::shared_ptr<Ort::Session> session;
private:
};

