// CPUQueryHandler.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "cpu_query_handler.h"
#include "iostream"
#include "vector"
#include "models.h"
#include "basic_cpu_processor.h"
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

void RunCPULineItems(std::vector<LineItem>& items) {
	std::cout << "Running line items CPU processor\n";
	BasicCPUProcessor<LineItem> processor(items);
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = processor.Filter(&LineItemFilter);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n\n";

	delete &results;
}

void RunCPUOrders(std::vector<Order>& orders) {
	std::cout << "Running orders CPU processor\n";
	BasicCPUProcessor<Order> processor(orders);
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
	double duration;
	std::clock_t start = std::clock();
	if (query == Query::SIMPLE_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");

		std::cout << "Done reading\n";
		duration = GetElapsedTime(start);
		std::cout << "Reading took " << duration << "ms\n\n";

		RunCPUOrders(orders);
		delete &orders;
	}
	else if (query == Query::SIMPLE_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		std::cout << "Done reading\n";

		duration = GetElapsedTime(start);
		std::cout << "Reading took " << duration << "ms\n\n";

		RunCPULineItems(items);
		delete &items;
	}
}
