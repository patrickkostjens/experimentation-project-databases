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

void RunCPU(std::vector<LineItem>& items) {
	std::cout << "Running CPU processor\n";
	BasicCPUProcessor<LineItem> processor(items);
	std::clock_t start = std::clock();
	int resultCount = processor.Filter(&LineItemFilter).size();

	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n";
}

void RunGPU(std::vector<LineItem>& items) {
	std::cout << "Running GPU processor\n";
	std::clock_t start = std::clock();
	gpu_filter(items);
	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "GPU processing took " << duration << "ms\n";	
}

int _tmain(int argc, TCHAR* argv[]) {
	std::clock_t total_start = std::clock();
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	double duration;
	std::clock_t start = std::clock();

	std::vector<LineItem> items = ReadAllLineItems("..\\..\\lineitem.tbl");
	std::cout << "Done reading\n";

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Reading took " << duration << "ms\n";

	if (options.processing_mode == ProcessingMode::CPU) {
		RunCPU(items);
	}
	else if (options.processing_mode == ProcessingMode::GPU) {
		RunGPU(items);
	}
	else if (options.processing_mode == ProcessingMode::ALL) {
		RunCPU(items);
		RunGPU(items);
	}
	
	double total_duration = (std::clock() - total_start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Total took " << total_duration << "ms\n";

	std::cin.get();
	return 0;
}
