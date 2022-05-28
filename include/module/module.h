#pragma once
#include <string>
#include <stdio.h>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "onnx/onnxutils.h"

class Module
{
public:
	Module();
	Module(Module& m) = delete;
	Module& operator=(Module& m) = delete;
	virtual ~Module();
	// env,session options继承自模板类base，base delete 复制构造和operator =,没有使用构造函数一级一级传递，
	void Init(Ort::Env&,Ort::SessionOptions&,const wchar_t * model_path, Ort::AllocatorWithDefaultOptions& allocator);
protected:
	onnxutils::ModelIOInfo model_info;
	std::shared_ptr<Ort::Session> session;
private:

};

