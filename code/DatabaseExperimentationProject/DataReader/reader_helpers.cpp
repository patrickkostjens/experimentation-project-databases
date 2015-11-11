#include "stdafx.h"
#include <fstream>
#include "reader_helpers.h"

const tm ParseDate(const std::string& date_string) {
	time_t raw_time;
	time(&raw_time);
	struct tm date;
	localtime_s(&date, &raw_time);
	date.tm_year = atoi(date_string.substr(0, 4).c_str());
	date.tm_mon = atoi(date_string.substr(5, 2).c_str()) - 1;
	date.tm_mday = atoi(date_string.substr(8, 2).c_str());

	return date;
}

void ReadUntilSeparator(std::ifstream& in_file, char* out, const int len) {
	std::string buf;
	std::getline(in_file, buf, '|');

	strcpy_s(out, len, buf.c_str());
}

const tm ReadDate(std::ifstream& in_file) {
	std::string buf;
	std::getline(in_file, buf, '|');

	return ParseDate(buf);
}
