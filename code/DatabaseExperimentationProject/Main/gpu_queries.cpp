#include "stdafx.h"
#include <iostream>
#include "models.h"
#include "basic_gpu_filter.h"
#include "unified_memory_gpu_filter.h"
#include "async_gpu_filter.h"
#include "helpers.h"

void RunStandardFilter(std::vector<LineItem>& items) {
	std::cout << "Running line items GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = gpu_filter(items);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunUnifiedMemoryFilter(std::vector<LineItem>& items) {
	std::cout << "Running line items GPU Unified Memory processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = um_gpu_filter(items);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU Unified Memory result count: " << resultCount << "\n";
	std::cout << "GPU Unified Memory processing took " << duration << "ms\n\n";

	delete &results;
}

void RunAsyncFilter(std::vector<LineItem>& items) {
	std::cout << "Running line items GPU Async processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = gpu_filter_async(items);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU Async result count: " << resultCount << "\n";
	std::cout << "GPU Async processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<LineItem>& items) {
	RunStandardFilter(items);
	// First run takes significantly longer, so run twice
	RunStandardFilter(items);
	RunUnifiedMemoryFilter(items);
	RunAsyncFilter(items);
}

void RunStandardFilter(std::vector<Order>& orders) {
	std::cout << "Running orders GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = gpu_filter(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunUnifiedMemoryFilter(std::vector<Order>& orders) {
	std::cout << "Running orders GPU UnifiedMemory processor\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = um_gpu_filter(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunAsyncFilter(std::vector<Order>& orders) {
	std::cout << "Running orders GPU async processor\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = gpu_filter_async(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<Order>& orders) {
	RunStandardFilter(orders);
	// First run takes significantly longer, so run twice
	RunStandardFilter(orders);
	RunUnifiedMemoryFilter(orders);
	RunAsyncFilter(orders);
}
