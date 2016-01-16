// Read lineitem table as generated by TPC-H

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "line_item_parser.h"
#include "reader_helpers.h"
#include <algorithm>


std::vector<LineItem>& ReadAndParseLineItems(const std::string& file_path) {
	std::vector<LineItem>& items = *new std::vector<LineItem>;
	
	std::ifstream in_file(file_path);

	if (!in_file.good()) {
		std::cerr << "File '" << file_path << "' does not exist\n";
	}

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
		&& counter != 100000) {
		current.ship_date = ReadDate(in_file);
		current.commit_date = ReadDate(in_file);
		current.receipt_date = ReadDate(in_file);

		ReadUntilSeparator(in_file, current.ship_instruct, sizeof(current.ship_instruct) / sizeof(current.ship_instruct[0]));
		ReadUntilSeparator(in_file, current.ship_mode, sizeof(current.ship_mode) / sizeof(current.ship_mode[0]));
		ReadUntilSeparator(in_file, current.comment, sizeof(current.comment) / sizeof(current.comment[0]));
		items.push_back(current);
		counter++;
		if (counter % 1000 == 0) {
			std::cout << "Line items read: " << counter << "\r";
		}
	}
	// Include some spaces to overwrite big numbers that might be printed above
	std::cout << "Done reading line items     \n";
	std::random_shuffle(items.begin(), items.end(), random);
	return items;
}
