// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "data_reader.h"
#include "line_item_parser.h"
#include "order_parser.h"

std::vector<LineItem>& ReadAllLineItems(const std::string file_path) {
	return ReadAndParseLineItems(file_path);
}

std::vector<Order>& ReadAllOrders(const std::string file_path) {
	return ReadAndParseOrders(file_path);
}
