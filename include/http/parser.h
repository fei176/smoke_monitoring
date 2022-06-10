#pragma once
#include<string>
#include<memory>
#include<iostream>

void strip(std::string& str);

class Parser {
public:
	Parser(std::string&,std::shared_ptr<std::string>);
	void parse();
private:
	std::string boundary;
	std::shared_ptr < std::string> data;

	int cur_pos;
	int line_start;
	int line_length;

	std::string content_type;
	std::string name;
	std::string filename;
	std::string value;
	int start_pos;
	int length;

	bool boundary_end;

	bool nextline();
	bool atBoundaryLine();
	bool atEndData();
	void parseHeader();
	void parseData();
	std::string getDispositionValue(const std::string& line, int start_index, std::string );
};