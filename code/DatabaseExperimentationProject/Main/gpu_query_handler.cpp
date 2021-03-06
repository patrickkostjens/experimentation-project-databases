#include "stdafx.h"
#include "../DataReader/data_reader.h"
#include "../Utils/command_line_options.h"
#include <iostream>
#include "gpu_queries.h"

void ExecuteGPUQuery(const Query& query) {
	if (query == Query::FILTER_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunGPUFilter(items);
		delete &items;
	}
	else if (query == Query::FILTER_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunGPUFilter(orders);
		delete &orders;
	}
	else if (query == Query::JOIN_LINE_ITEM_ORDERS) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunGPUSortMergeJoin(items, orders);
		delete &items;
		delete &orders;
	}
	else {
		std::cerr << "GPU: Unsupported query\n";
	}
}
