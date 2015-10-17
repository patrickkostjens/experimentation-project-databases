// MainCPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "data_reader.h"
#include "basic_cpu_processor.h"
#include "ctime"


bool LineItemFilter(const LineItem item) {
	return item.order_key == 1;
}


int _tmain(const int argc, const TCHAR* argv[]) {
	std::clock_t total_start = std::clock();
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	double duration;
	std::clock_t start = std::clock();

	std::vector<LineItem>& items = ReadAllLineItems("..\\..\\lineitem.tbl");
	std::cout << "Done reading\n";

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Reading took " << duration << "ms\n";

	BasicCPUProcessor<LineItem> processor(items);
	start = std::clock();
	int resultCount = processor.Filter(&LineItemFilter).size();

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "CPU result count: " << resultCount << "\n";
	std::cout << "CPU Filtering took " << duration << "ms\n";

	double total_duration = (std::clock() - total_start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Total took " << total_duration << "ms\n";

	std::cin.get();
	return 0;
}