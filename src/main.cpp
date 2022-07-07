#include "module/modelzoo.h"
#include "http/sserver.h"
#include <iostream>
#ifdef _WIN32
#include <direct.h>
#elif __APPLE__ || __linux__
#include<unistd.h>
#endif

int main(int argc,char * argv[]) {
	std::cout << "input format:"<< std::endl;
	std::cout << "    ip:           port:   /path/to/web:   /weights/path:" << std::endl;
	std::cout << "eg: 192.168.1.1   8888    c:/web          c:/weights" << std::endl;
	std::cout << "or input nothing" << std::endl;
	std::string ip;
	std::string port;
	std::string doc_path;
	std::string weight_path;
	if (argc == 5) {
		ip = std::string(argv[1]);
		port = std::string(argv[2]);
		doc_path = std::string(argv[3]) + "/web";
		weight_path = std::string(argv[4]) + "/weights";
	}
	else if (argc == 1) {
		ip = std::string("127.0.0.1");
		port = std::string("6573");
		char runpath[1024] = { 0 };
		#ifdef _WIN32
		_getcwd(runpath, sizeof(runpath));
		#elif __APPLE__ || __linux__
		getcwd(runpath, sizeof(runpath));
		#endif
		doc_path = std::string(runpath) + "/web";
		weight_path = std::string(runpath) + "/weights";
	}
	else {
		std::cout << "input format:" << std::endl;
		std::cout << "    ip:           port:   /path/to/web:   /weights/path:" << std::endl;
		std::cout << "eg: 192.168.1.1   8888    c:/web          c:/weights" << std::endl;
		std::cout << "or input nothing" << std::endl;
		return 0;
	}
	std::unique_ptr<ModelZoo> zoo(ModelZoo::getInstanse(weight_path.c_str()));
	TThreadPool pool(8,4);
	pool.Start();
	boost::asio::io_context ico;
	pool.Push(
		[&ico,&pool,doc_path, ip,port]
		{
			try {
				SServer server(ip, std::stoi(port), ico, pool, doc_path);
				server.run();
			}
			catch(...)
			{	
				std::cout << "a bug" << std::endl;
				return;
			}
		});
	char c;
	while (c=getchar()) {
		if (c == 'q') {
			pool.Stop();
			ico.stop();
			break;
		}
		if (c == '1') {
			std::cout << pool.RunningCount() << std::endl;
		}
	}
	return 0;
}