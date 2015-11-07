#include "stdafx.h"
#include <iostream>
#include "models.h"
#include "basic_gpu_filter.h"
#include "unified_memory_gpu_filter.h"
#include "async_gpu_filter.h"
#include "helpers.h"

template<typename Filter, typename Item>
void RunGenericFilter(char* filterName, char* typeName, std::vector<Item>& orders, Filter& filterLambda) {
	std::cout << "Running " << typeName << " " << filterName << "\n";
	std::clock_t start = std::clock();
	std::vector<Item>& results = filterLambda(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<LineItem>& items) {
	RunGenericFilter("standard GPU processor", "line item", items, gpu_filter<LineItem>);
	// First run takes significantly longer, so run twice
	RunGenericFilter("standard GPU processor", "line item", items, gpu_filter<LineItem>);
	RunGenericFilter("Unified Memory GPU processor", "line item", items, um_gpu_filter<LineItem>);
	RunGenericFilter("async GPU processor", "line item", items, gpu_filter_async<LineItem>);
}

void RunGPUFilter(std::vector<Order>& orders) {
	RunGenericFilter("standard GPU processor", "orders", orders, gpu_filter<Order>);
	// First run takes significantly longer, so run twice
	RunGenericFilter("standard GPU processor", "orders", orders, gpu_filter<Order>);
	RunGenericFilter("Unfied Memory GPU processor", "orders", orders, um_gpu_filter<Order>);
	RunGenericFilter("async GPU processor", "orders", orders, gpu_filter_async<Order>);
}
