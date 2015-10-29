// CPUQueryHandler.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "cpu_query_handler.h"
#include "iostream"
#include "vector"
#include "models.h"
#include "basic_cpu_filter.h"
#include "data_reader.h"

bool LineItemFilter(LineItem item) {
	return item.order_key == 1;
}

bool OrderFilter(Order order) {
	return order.order_status == 'O';
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
}
