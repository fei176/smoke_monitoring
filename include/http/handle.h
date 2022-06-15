#pragma once
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "module/modelzoo.h"

class Handel {
public:
	Handel(std::string doc_root){
		this->doc_root = doc_root;
	};
	template<class Body, class Allocator, class Send>
	void response(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>&& req,Send&& send);
protected:
	boost::beast::string_view get_type(boost::beast::string_view);
	std::string path_cat(boost::beast::string_view base, boost::beast::string_view path);

	template<class Body, class Allocator>
    boost::beast::http::response<boost::beast::http::string_body> bad_request(boost::beast::string_view why, 
        boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>& req) {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::bad_request, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = std::string(why);
        res.prepare_payload();
        return res;
    }
	template<class Body, class Allocator>
    boost::beast::http::response<boost::beast::http::string_body> not_found(boost::beast::string_view why, boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>& req) {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::not_found, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "The resource was not found.";
        res.prepare_payload();
        return res;
    }
	template<class Body, class Allocator>
    boost::beast::http::response<boost::beast::http::string_body> server_error(boost::beast::string_view why, boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>& req) {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::internal_server_error, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "An error occurred: '" + std::string(why) + "'";
        res.prepare_payload();
        return res;
    }
private:
	std::string doc_root;
};


template<class Body, class Allocator, class Send>
void Handel::response(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>&& req, Send&& send) {
    //if (req.method() != boost::beast::http::verb::get &&
    //    req.method() != boost::beast::http::verb::head)
    //    return send(std::move(bad_request("Unknown HTTP-method", req)));

    if (req.method() == boost::beast::http::verb::get) {
        if (req.target().empty() ||
            req.target()[0] != '/' ||
            req.target().find("..") != boost::beast::string_view::npos)
            return  send(std::move(bad_request("Illegal request-target", req)));

        std::string path = path_cat(doc_root, req.target());
        if (req.target().back() == '/')
            path.append("index.html");

        boost::beast::error_code ec;
        boost::beast::http::file_body::value_type body;
        body.open(path.c_str(), boost::beast::file_mode::scan, ec);

        if (ec == boost::beast::errc::no_such_file_or_directory)
            return send(std::move(not_found(req.target(), req)));

        if (ec)
            return send(std::move(server_error(ec.message(), req)));

        auto const size = body.size();
        boost::beast::http::response<boost::beast::http::file_body> res{
        std::piecewise_construct,
        std::make_tuple(std::move(body)),
        std::make_tuple(boost::beast::http::status::ok, req.version()) };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, get_type(path));
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }
    else if (req.method() == boost::beast::http::verb::post) {
        auto body = req.body();

        char* all_data = (char*)(body.data());
        int padding = int(*(all_data + 20));
        float* par = (float*)all_data;
        std::vector<uchar> img_data(body.begin()+ padding + 20, (body.end()));
        cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);

        int id = par[0];
        int h = par[1];
        int w = par[2];
        float nms = par[3];
        float conf = par[4];
        ModelZoo* zoo = ModelZoo::getInstanse();
        cv::Mat result = zoo->forward(id, h, w, nms, conf,img);

        cv::imencode(".png", result, img_data);
        boost::beast::http::vector_body<uchar> return_body;
        auto const size = return_body.size(img_data);
    
        boost::beast::http::response<boost::beast::http::vector_body<uchar>> res{
            std::piecewise_construct,
            std::make_tuple(std::move(img_data)),
            std::make_tuple(boost::beast::http::status::ok, req.version()) };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "image/png");
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }
    else {
        boost::beast::http::response<boost::beast::http::empty_body> res{ boost::beast::http::status::ok, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.content_length(0);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }
}