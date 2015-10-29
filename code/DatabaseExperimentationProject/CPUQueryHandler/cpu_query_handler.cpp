// CPUQueryHandler.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <tuple>
#include "cpu_query_handler.h"
#include "iostream"
#include "vector"
#include "models.h"
#include "basic_cpu_filter.h"
#include "data_reader.h"
#include "cpu_sort_merge_join.h"

bool LineItemFilter(LineItem item) {
	return item.order_key == 1;
}

bool OrderFilter(Order order) {
	return order.order_status == 'O';
}

int LineItemJoinPropertySelector(LineItem item) {
	return item.order_key;
}

int OrderJoinPropertySelector(Order order) {
	return order.order_key;
}

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

void RunCPUFilter(std::vector<LineItem>& items) {
	std::cout << "Running line items CPU filter\n";
	BasicCPUFilter<LineItem> processor(items);
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = processor.Filter(&LineItemFilter);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n\n";

	delete &results;
}

void RunCPUFilter(std::vector<Order>& orders) {
	std::cout << "Running orders CPU filter\n";
	BasicCPUFilter<Order> processor(orders);
	std::clock_t start = std::clock();
	std::vector<Order>& results = processor.Filter(&OrderFilter);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n\n";

	delete &results;
}

void RunCPUSortMergeJoin(std::vector<LineItem>& items, std::vector<Order>& orders) {
	std::cout << "Running CPU sort merge join\n";
	CPUSortMergeJoin<Order, LineItem> processor(orders, items);
	std::clock_t start = std::clock();
	std::vector<std::tuple<Order, LineItem>>& results = processor.Join(&OrderJoinPropertySelector, &LineItemJoinPropertySelector);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU joining took " << duration << "ms\n\n";

	delete &results;
}

void ExecuteCPUQuery(Query query)
{
	if (query == Query::SIMPLE_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunCPUFilter(orders);
		delete &orders;
	}
	else if (query == Query::SIMPLE_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunCPUFilter(items);
		delete &items;
	}
	else if (query == Query::SORT_MERGE_JOIN_LINE_ITEM_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunCPUSortMergeJoin(items, orders);
		delete &items;
		delete &orders;
	}
}
