// Read orders table as generated by TPC-H

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "order_parser.h"
#include "reader_helpers.h"
#include <algorithm>

std::vector<Order>& ReadAndParseOrders(const std::string& file_path) {
	std::vector<Order>& orders = *new std::vector<Order>;

	std::ifstream in_file(file_path);

	if (!in_file.good()) {
		std::cerr << "File '" << file_path << "' does not exist\n";
	}

	int counter = 0;

	Order current;
	char delimiter;
	while (in_file >> current.order_key >> delimiter
		>> current.customer_key >> delimiter
		>> current.order_status >> delimiter
		>> current.total_price >> delimiter
		&& counter != 1000) {
		current.order_date = ReadDate(in_file);

		ReadUntilSeparator(in_file, current.order_priority, sizeof(current.order_priority) / sizeof(current.order_priority[0]));
		ReadUntilSeparator(in_file, current.clerk, sizeof(current.clerk) / sizeof(current.clerk[0]));
		in_file >> current.ship_priority >> delimiter;
		ReadUntilSeparator(in_file, current.comment, sizeof(current.comment) / sizeof(current.comment[0]));

		orders.push_back(current);
		counter++;
		if (counter % 1000 == 0) {
			std::cout << "Orders read: " << counter << "\r";
		}
	}
	// Include some spaces to overwrite big numbers that might be printed above
	std::cout << "Done reading orders     \n";
	std::random_shuffle(orders.begin(), orders.end(), random);

	return orders;
}
