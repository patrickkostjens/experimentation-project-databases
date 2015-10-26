// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "data_reader.h"
#include "basic_gpu_filter.h"
#include "ctime"
#include "cpu_query_handler.h"

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

void RunGPU(std::vector<LineItem>& items) {
	std::cout << "Running line items GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = gpu_filter(items);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";	

	delete &results;
}

void RunGPU(std::vector<Order>& orders) {
	std::cout << "Running orders GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<Order>& results = gpu_filter(orders);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";

	delete &results;
}

void ExecuteGPUQuery(const Query& query) {
	if (query == Query::SIMPLE_LINE_ITEM) {
		std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
		RunGPU(items);
		delete &items;
	}
	else if (query == Query::SIMPLE_ORDERS) {
		std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
		RunGPU(orders);
		delete &orders;
	}
	else {
		std::cerr << "Unsupported query\n";
	}
}

int _tmain(const int argc, const TCHAR* argv[]) {
	std::clock_t total_start = std::clock();
	const CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	if (options.processing_mode == ProcessingMode::ALL || options.processing_mode == ProcessingMode::CPU) {
		ExecuteCPUQuery(options.query);
	}
	
	if (options.processing_mode == ProcessingMode::ALL || options.processing_mode == ProcessingMode::GPU) {
		ExecuteGPUQuery(options.query);
	}
	
	double total_duration = GetElapsedTime(total_start);
	std::cout << "Total took " << total_duration << "ms\n";

	std::cin.get();
	return 0;
}
