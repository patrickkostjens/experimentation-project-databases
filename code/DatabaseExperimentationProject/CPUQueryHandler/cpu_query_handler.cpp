#include "stdafx.h"
#include <tuple>
#include "cpu_query_handler.h"
#include "iostream"
#include "vector"
#include "../Models/models.h"
#include "../BasicCPUProcessor/basic_cpu_filter.h"
#include "../DataReader/data_reader.h"
#include "../BasicCPUProcessor/cpu_sort_merge_join.h"
#include "../BasicCPUProcessor/cpu_hash_join.h"
#include "../BasicCPUProcessor/indexed_cpu_filter.h"

bool LineItemFilter(LineItem item) {
	return item.extended_price < 20000;
}

int LineItemFilterPropertySelector(LineItem item) {
	return item.order_key;
}

bool OrderFilter(Order order) {
	return order.order_status == 'O';
}

char OrderFilterPropertySelector(Order order) {
	return order.order_status;
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
#if DEBUG
	std::cout << "Running line items CPU filter\n";
#endif
	BasicCPUFilter<LineItem> processor(items);
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = processor.Filter(&LineItemFilter);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
#if DEBUG
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n\n";
#endif
	std::cout << duration << "\n";

	delete &results;
}

void RunCPUFilter(std::vector<Order>& orders) {
#if DEBUG
	std::cout << "Running orders CPU filter\n";
#endif
	BasicCPUFilter<Order> processor(orders);
	std::clock_t start = std::clock();
	std::vector<Order>& results = processor.Filter(&OrderFilter);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
#if DEBUG
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n\n";
#endif
	std::cout << duration << "\n";

	delete &results;
}

void RunIndexedCPUFilter(std::vector<LineItem>& items) {
#if DEBUG
	std::cout << "Running indexed line items CPU filter\n";
#endif
	IndexedCPUFilter<LineItem, int> processor(items);
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = processor.Filter(&LineItemFilterPropertySelector, 1);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
#if DEBUG
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Indexed filtering took " << duration << "ms\n\n";
#endif

	delete &results;
}

void RunIndexedCPUFilter(std::vector<Order>& orders) {
#if DEBUG
	std::cout << "Running indexed orders CPU filter\n";
#endif
	IndexedCPUFilter<Order, char> processor(orders);
	std::clock_t start = std::clock();
	std::vector<Order>& results = processor.Filter(&OrderFilterPropertySelector, 'O');
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
#if DEBUG
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Indexed filtering took " << duration << "ms\n\n";
#endif

	delete &results;
}

void RunCPUSortMergeJoin(std::vector<LineItem>& items, std::vector<Order>& orders) {
#if DEBUG
	std::cout << "Running CPU sort merge join\n";
#endif
	CPUSortMergeJoin<Order, LineItem> processor(orders, items);
	std::clock_t start = std::clock();
	std::vector<std::tuple<Order, LineItem>>& results = processor.Join(&OrderJoinPropertySelector, &LineItemJoinPropertySelector);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << duration << "\n";
#if DEBUG
	std::cout << "CPU sort-merge join result count: " << resultCount << "\n";
	std::cout << "CPU sort-merge joining took " << duration << "ms\n\n";
#endif

	delete &results;
}

void RunCPUHashJoin(std::vector<LineItem>& items, std::vector<Order>& orders) {
#if DEBUG
	std::cout << "Running CPU hash join\n";
#endif
	CPUHashJoin<Order, LineItem> processor(orders, items);
	std::clock_t start = std::clock();
	std::vector<std::tuple<Order, LineItem>>& results = processor.Join(&OrderJoinPropertySelector, &LineItemJoinPropertySelector);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << duration << "\n";
#if DEBUG
	std::cout << "CPU hash join result count: " << resultCount << "\n";
	std::cout << "CPU hash joining took " << duration << "ms\n\n";
#endif

	delete &results;
}

void ExecuteCPUQuery(Query query)
{
	if (query == Query::FILTER_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunCPUFilter(orders);
		for (int i = 0; i < 10; i++) {
			RunCPUFilter(orders);
		}
		delete &orders;
	}
	else if (query == Query::FILTER_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunCPUFilter(items);
		for (int i = 0; i < 10; i++) {
			RunCPUFilter(items);
		}
		delete &items;
	}
	else if (query == Query::INDEXED_FILTER_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");

		RunIndexedCPUFilter(items);
		for (int i = 0; i < 10; i++) {
			RunIndexedCPUFilter(items);
		}
		delete &items;
	}
	else if (query == Query::INDEXED_FILTER_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunIndexedCPUFilter(orders);
		for (int i = 0; i < 10; i++) {
			RunIndexedCPUFilter(orders);
		}
		delete &orders;
	}
	else if (query == Query::JOIN_LINE_ITEM_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunCPUSortMergeJoin(items, orders);
		for (int i = 0; i < 10; i++) {
			RunCPUSortMergeJoin(items, orders);
		}
		std::cout << "---";
		RunCPUHashJoin(items, orders);
		for (int i = 0; i < 10; i++) {
			RunCPUHashJoin(items, orders);
		}
		delete &items;
		delete &orders;
	}
}
