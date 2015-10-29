#include "stdafx.h"
#include "iostream"
#include "models.h"
#include "basic_gpu_filter.h"
#include "helpers.h"

void RunGPUFilter(std::vector<LineItem>& items) {
	std::cout << "Running line items GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = gpu_filter(items);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void RunGPUFilter(std::vector<Order>& orders) {
	std::cout << "Running orders GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = gpu_filter(orders);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}
