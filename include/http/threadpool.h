#pragma once
#include <future>
#include <queue>
class TThreadPool {
public:
	TThreadPool();

	TThreadPool(int thread_nums, int extra_thread_nums);

	// 线程池的入口函数，启动线程池。
	void Start() ;

	void Run() ;

	int RunningCount() ;

	void Stop();

	bool IsExit() ;

	template<class T, class... Types>
	auto Push(T&& f, Types&&... args)
#ifdef C++
		->std::future<typename std::invoke_result<T, Types...>::type>;
#else
		->std::future<typename std::result_of<T(Types...)>::type>;
#endif

	std::function<void()> Pop() ;
private:
	bool is_start;
	int thread_nums;
	int extra_thread_nums;
	bool is_exit;

	std::atomic<int> runing_count = 0;

	std::mutex mut;
	std::condition_variable cv_task;

	std::vector<std::shared_ptr<std::thread>> threads;
	std::queue< std::function<void()> > tasks;
};

template<class T, class... Types>
auto TThreadPool::Push(T&& f, Types&&... args)
#ifdef C++
->std::future<typename std::invoke_result<T, Types...>::type>
#else
->std::future<typename std::result_of<T(Types...)>::type>
#endif
{
	{
		// 当线程够的时候，直接加入
		// 不够，备用够
		// 都不够
		std::unique_lock<std::mutex> lock(mut);
#ifdef C++
		using return_type = typename std::invoke_result<T, Types...>::type;
#else
		using return_type = typename std::result_of<T(Types...)>::type;
#endif
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