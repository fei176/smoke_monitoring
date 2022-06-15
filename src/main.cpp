#pragma once
#include "module/modelzoo.h"
#include "http/server.h"
#include "http/sserver.h"
#include <iostream>
#ifdef _WIN32
#include <direct.h>
#elif __APPLE__ || __linux__
#include<unistd.h>
#endif


int main(int argc,char * argv[]) {
	std::cout << "input    IP:          port:  /path/to/main/web/dir" << std::endl;
	std::cout << "     eg: 192.168.1.1  8888   c:/web" << std::endl;
	std::cout << "     or input nothing" << std::endl;
	std::string ip;
	std::string port;
	std::string doc_path;
	if (argc == 4) {
		ip = std::string(argv[1]);
		port = std::string(argv[2]);
		doc_path = std::string(argv[3]);
	}
	else if (argc == 1) {
		ip = std::string("127.0.0.1");
		port = std::string("6573");
		char runpath[1024] = { 0 };
		_getcwd(runpath, sizeof(runpath));
		doc_path = std::string(runpath) + "/web";
	}
	else {
		std::cout << "input    IP:          port:  /path/to/main/web/dir" << std::endl;
		std::cout << "     eg: 192.168.1.1  8888   c:/web" << std::endl;
		std::cout << "         or input nothing" << std::endl;
		return 0;
	}
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
			ico.stop();
			pool.Stop();
			break;
		}
		if (c == '1') {
			std::cout << pool.RunningCount() << std::endl;
		}
	}
	return 0;
}