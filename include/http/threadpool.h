#pragma once
#include <future>
#include <queue>
#include <functional>

class TThreadPool {
public:
	TThreadPool();

	TThreadPool(int thread_nums, int extra_thread_nums);

	// what?
	void Start() ;

	void Run() ;

	int RunningCount() ;

	void Stop();

	bool IsExit() ;

	template<class T, class... Types>
	auto Push(T&& f, Types&&... args)
		->std::future<typename std::result_of<T(Types...)>::type>;

	std::function<void()> Pop() ;
private:
	bool is_start;
	int thread_nums;
	int extra_thread_nums;
	bool is_exit;

	std::atomic<int> runing_count;

	std::mutex mut;
	std::condition_variable cv_task;

	std::vector<std::shared_ptr<std::thread>> threads;
	std::queue< std::function<void()> > tasks;
};

template<class T, class... Types>
auto TThreadPool::Push(T&& f, Types&&... args)->std::future<typename std::result_of<T(Types...)>::type>
{
	{
		std::unique_lock<std::mutex> lock(mut);
		using return_type = typename std::result_of<T(Types...)>::type;
		auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<T>(f), std::forward<Types>(args)...));
		std::future<return_type> result = task->get_future();
		tasks.emplace([task]() {
			(*task)(); }
		);
		lock.unlock();
		cv_task.notify_one();
		return result;
	}
}