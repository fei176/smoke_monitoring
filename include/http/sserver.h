#pragma once

#include <string>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "http/handle.h"
#include "http/threadpool.h"

class SServer {
public:
	SServer(std::string ip, short port, boost::asio::io_context&,TThreadPool&,std::string);
	~SServer();

	template<class Stream>
	struct Send_impl {
		Stream& stream;
		boost::beast::error_code& ec;
		bool& close;
		Send_impl(boost::beast::error_code& ec_, bool& close_, Stream& stream_)
			:ec{ ec_ }, close{ close_ }, stream{ stream_ } {

		}
		template<bool isRequest, class Body, class Fields>
		void operator()(boost::beast::http::message<isRequest, Body, Fields>&& msg) const
		{
			close = msg.need_eof();
			boost::beast::http::serializer<isRequest, Body, Fields> req{ msg };
			boost::beast::http::write(stream, req, ec);
		}
	};

	void run();
	void session(boost::asio::ip::tcp::socket& socket);
private:
	void fail(boost::beast::error_code ec, char const* what);
	std::string doc_root;
	TThreadPool& pool;
	boost::asio::io_context& ico;
	boost::asio::ip::tcp::acceptor net_acceptor;
};