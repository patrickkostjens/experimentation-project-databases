// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "data_reader.h"
#include "basic_cpu_processor.h"
#include "basic_gpu_filter.h"
#include "ctime"


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

void RunGPU(std::vector<LineItem>& items) {
	std::cout << "Running GPU processor\n";
	std::clock_t start = std::clock();
	std::vector<LineItem>& results = gpu_filter(items);
	int resultCount = results.size();

	double duration = GetElapsedTime(start);
	std::cout << "GPU result count: " << resultCount << "\n";
	std::cout << "GPU processing took " << duration << "ms\n\n";	

	delete &results;
}

int _tmain(const int argc, const TCHAR* argv[]) {
	std::clock_t total_start = std::clock();
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	double duration;
	std::clock_t start = std::clock();

	std::vector<Order>& orders = ReadAllOrders("..\\..\\orders.tbl");
	std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
	std::cout << "Done reading\n";

	duration = GetElapsedTime(start);
	std::cout << "Reading took " << duration << "ms\n\n";

	if (options.processing_mode == ProcessingMode::CPU) {
		RunCPULineItems(items);
		RunCPUOrders(orders);
	}
	else if (options.processing_mode == ProcessingMode::GPU) {
		RunGPU(items);
	}
	else if (options.processing_mode == ProcessingMode::ALL) {
		RunCPULineItems(items);
		RunCPUOrders(orders);
		RunGPU(items);
	}
	
	double total_duration = GetElapsedTime(total_start);
	std::cout << "Total took " << total_duration << "ms\n";

	// Cleanup
	delete &items;
	delete &orders;

	std::cin.get();
	return 0;
}
