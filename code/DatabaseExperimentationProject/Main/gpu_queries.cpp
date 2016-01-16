#include "stdafx.h"
#include <iostream>
#include "../Models/models.h"
#include "../BasicGPUProcessor/basic_gpu_filter.h"
#include "../BasicGPUProcessor/unified_memory_gpu_filter.h"
#include "../BasicGPUProcessor/async_gpu_filter.h"
#include "../BasicGPUProcessor/gpu_sort_merge_join.h"
#include "helpers.h"
#include <tuple>

template<typename Filter, typename Item>
void RunGenericFilter(char* filterName, char* typeName, std::vector<Item>& orders, Filter& filterLambda, bool printResultCount) {
#if DEBUG
	std::cout << "Running " << typeName << " " << filterName << "\n";
#endif
	std::clock_t start = std::clock();
	std::vector<Item>& results = filterLambda(orders);
	size_t resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << duration << "\n";
	if (printResultCount) {
		std::cout << "GPU result count: " << resultCount << "\n";
	}
#if DEBUG
	std::cout << "GPU processing took " << duration << "ms\n\n";
#endif

	delete &results;
}

void RunGPUFilter(std::vector<LineItem>& items) {
	RunGenericFilter("standard GPU processor", "line item", items, gpu_filter<LineItem>, true);
	// First run takes significantly longer, so run twice
	for (int i = 0; i < 10; i++) {
		//RunGenericFilter("standard GPU processor", "line item", items, gpu_filter<LineItem>, false);
		RunGenericFilter("Unified Memory GPU processor", "line item", items, um_gpu_filter<LineItem>, false);
		//RunGenericFilter("async GPU processor", "line item", items, gpu_filter_async<LineItem>, false);
	}
}

void RunGPUFilter(std::vector<Order>& orders) {
	RunGenericFilter("standard GPU processor", "orders", orders, gpu_filter<Order>, true);
	// First run takes significantly longer, so run twice
	for (int i = 0; i < 10; i++) {
		//RunGenericFilter("standard GPU processor", "orders", orders, gpu_filter<Order>, false);
		RunGenericFilter("Unfied Memory GPU processor", "orders", orders, um_gpu_filter<Order>, false);
		//RunGenericFilter("async GPU processor", "orders", orders, gpu_filter_async<Order>, false);
	}
}

void GPUSortMergeJoin(std::vector<LineItem>& items, std::vector<Order>& orders) {
	std::cout << "Running GPU Sort-Merge Join\n";
	std::clock_t start = std::clock();
	std::vector<std::tuple<Order, LineItem>>& results = gpu_sort_merge_join<Order, LineItem>(orders, items);

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << results.size() << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUSortMergeJoin(std::vector<LineItem>& items, std::vector<Order>& orders) {
	GPUSortMergeJoin(items, orders);
	GPUSortMergeJoin(items, orders);
}
