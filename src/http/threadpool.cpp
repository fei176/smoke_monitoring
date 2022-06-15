#include "http/threadpool.h"

TThreadPool::TThreadPool(){
	this->thread_nums = 16;
	this->extra_thread_nums = 8;
	is_start = false;
	is_exit = false;
}

TThreadPool::TThreadPool(int thread_nums, int extra_thread_nums) {
	this->thread_nums = thread_nums;
	this->extra_thread_nums = extra_thread_nums;
	is_start = false;
	is_exit = false;
}

void TThreadPool::Start() {
	if (is_start) {
		return;
	}
	std::unique_lock<std::mutex> lock(mut);
	is_start = true;
	for (int i = 0; i < thread_nums; i++) {
		/*thread* t = new thread(&TThreadPool::Run, this);*/
		std::shared_ptr<std::thread> th = std::make_shared<std::thread>(&TThreadPool::Run, this);
		threads.push_back(th);
	}
};

void TThreadPool::Run() {
	//“Ï≤Ω
	while (!is_exit) {
		std::function<void()> task = Pop();
		if (!is_exit) {
			runing_count += 1;
			task();
			runing_count -= 1;
		}
	}
};

int TThreadPool::RunningCount() {
	return runing_count;
};

void TThreadPool::Stop() {
	is_exit = true;
	cv_task.notify_all();
	for (auto& th : threads) {
		th->join();
	}
};

bool TThreadPool::IsExit() {
	return is_exit;
};

std::function<void()> TThreadPool::Pop() {
	std::unique_lock<std::mutex> lock(mut);
	if (tasks.empty()) {
		cv_task.wait(lock);
	}
	std::function<void()> next_task = nullptr;
	if (!tasks.empty()) {
		next_task = std::move(tasks.front());
		tasks.pop();
	}
	return next_task;
};