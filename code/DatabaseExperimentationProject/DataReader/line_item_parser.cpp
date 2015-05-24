#include "stdafx.h"
#include "line_item_parser.h"
#include "iostream"
#include "fstream"
#include "algorithm"
#include "time.h"
#include "string"

tm ParseDate(std::string date_string) {
	time_t raw_time;
	time(&raw_time);
	struct tm date;
	localtime_s(&date, &raw_time);
	date.tm_year = atoi(date_string.substr(0, 4).c_str());
	date.tm_mon = atoi(date_string.substr(5, 2).c_str()) - 1;
	date.tm_mday = atoi(date_string.substr(8, 2).c_str());

	return date;
}

void ReadUntilSeparator(std::ifstream* in_file, char* out, int len) {
	std::string buf;
	std::getline(*in_file, buf, '|');

	strcpy_s(out, len, buf.c_str());
}

tm ReadDate(std::ifstream* in_file) {
	std::string buf;
	std::getline(*in_file, buf, '|');

	return ParseDate(buf);
}

std::vector<LineItem> ReadAndParseLineItems(std::string file_path) {
	std::vector<LineItem> items;
	
	std::ifstream in_file(file_path);

	int counter = 0;

	LineItem current;
	char delimiter;
	while (in_file >> current.order_key >> delimiter
		>> current.part_key >> delimiter
		>> current.supp_key >> delimiter
		>> current.line_number >> delimiter
		>> current.quantity >> delimiter
		>> current.extended_price >> delimiter
		>> current.discount >> delimiter
		>> current.tax >> delimiter
		>> current.return_flag >> delimiter
		>> current.line_status >> delimiter
		&& counter != 10000) {
		current.ship_date = ReadDate(&in_file);
		current.commit_date = ReadDate(&in_file);
		current.receipt_date = ReadDate(&in_file);

		ReadUntilSeparator(&in_file, current.ship_instruct, sizeof(current.ship_instruct) / sizeof(current.ship_instruct[0]));
		ReadUntilSeparator(&in_file, current.ship_mode, sizeof(current.ship_mode) / sizeof(current.ship_mode[0]));
		ReadUntilSeparator(&in_file, current.comment, sizeof(current.comment) / sizeof(current.comment[0]));

		LineItem to_insert = current;
		items.push_back(to_insert);
		counter++;
		if (counter % 1000 == 0) {
			std::cout << counter << "\n";
		}
	}

	return items;
}
