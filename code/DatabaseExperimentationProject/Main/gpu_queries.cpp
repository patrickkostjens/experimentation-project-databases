#include "stdafx.h"
#include <iostream>
#include "models.h"
#include "basic_gpu_filter.h"
#include "unified_memory_gpu_filter.h"
#include "async_gpu_filter.h"
#include "helpers.h"

template<typename F>
void RunGenericLineItemFilter(char* name, std::vector<LineItem>& orders, F& filterLambda) {
	std::cout << "Running line items " << name << "\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = filterLambda(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<LineItem>& items) {
	RunGenericLineItemFilter("standard GPU processor", items, gpu_filter<LineItem>);
	// First run takes significantly longer, so run twice
	RunGenericLineItemFilter("standard GPU processor", items, gpu_filter<LineItem>);
	RunGenericLineItemFilter("Unified Memory GPU processor", items, um_gpu_filter<LineItem>);
	RunGenericLineItemFilter("async GPU processor", items, gpu_filter_async<LineItem>);
}

template<typename F>
void RunGenericOrderFilter(char* name, std::vector<Order>& orders, F& filterLambda) {
	std::cout << "Running orders " << name << "\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = filterLambda(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<Order>& orders) {
	RunGenericOrderFilter("standard GPU processor", orders, gpu_filter<Order>);
	// First run takes significantly longer, so run twice
	RunGenericOrderFilter("standard GPU processor", orders, gpu_filter<Order>);
	RunGenericOrderFilter("Unfied Memory GPU processor", orders, um_gpu_filter<Order>);
	RunGenericOrderFilter("async GPU processor", orders, gpu_filter_async<Order>);
}
