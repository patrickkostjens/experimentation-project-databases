#include "stdafx.h"
#include "data_reader.h"
#include "command_line_options.h"
#include <iostream>
#include "gpu_queries.h"

void ExecuteGPUQuery(const Query& query) {
	if (query == Query::SIMPLE_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunGPUFilter(items);
		delete &items;
	}
	else if (query == Query::SIMPLE_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunGPUFilter(orders);
		delete &orders;
	}
	else {
		std::cerr << "GPU: Unsupported query\n";
	}
}
