#include "http/sserver.h"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

SServer::SServer(std::string ip, short port, boost::asio::io_context& ico_, TThreadPool& pool_, std::string doc_root_)
	:ico{ ico_ }, net_acceptor{ net::make_strand(ico_) }, pool{ pool_ }{
	doc_root = (doc_root_);
	boost::system::error_code ec;
	auto const address = net::ip::make_address(ip, ec);
	if (ec) {
		fail(ec, "error address");
		return;
	}
	tcp::endpoint ep(address, port);
	net_acceptor.open(ep.protocol(), ec);
	if (ec) {
		fail(ec, "open");
		return;
	}
	net_acceptor.set_option(net::socket_base::reuse_address(true), ec);
	if (ec) {
		fail(ec, "set opention");
		return;
	}
	net_acceptor.bind(tcp::endpoint(address, port), ec);
	if (ec) {
		fail(ec, "bind");
		return;
	}
	net_acceptor.listen(net::socket_base::max_listen_connections, ec);
	if (ec) {
		fail(ec, "listen");
		return;
	}
}

SServer::~SServer() {
	std::cout << "release server" << std::endl;
}

void SServer::run() {
	for (;;)
	{
		tcp::socket socket{ ico };
		net_acceptor.accept(socket);
		std::thread{ std::bind(
				&SServer::session,
				this,
				std::move(socket)) }.detach();
		//pool.Push(&SServer::session,this,socket);
	}
}

void SServer::session(boost::asio::ip::tcp::socket& stream) {
	bool close = false;
	beast::error_code ec;
	beast::flat_buffer buffer;
	for (;;)
	{
		http::request<http::string_body> req;
		http::request_parser<http::string_body> parser;

		parser.body_limit(1024 * 1024 * 100);
		http::read(stream, buffer, parser, ec);
		if (ec == http::error::end_of_stream)
			break;
		if (ec)
			return fail(ec, "read");
		req = parser.release();

		Handel handel(doc_root);
		handel.response(std::move(req), Send_impl<tcp::socket>(ec, close, stream));
		if (ec)
			return fail(ec, "write");
		if (close)
		{
			break;
		}
		stream.shutdown(tcp::socket::shutdown_send, ec);
	}
}

void SServer::fail(boost::beast::error_code ec, char const* what) {
	std::cout << what << "; " << ec.message() << std::endl;
}