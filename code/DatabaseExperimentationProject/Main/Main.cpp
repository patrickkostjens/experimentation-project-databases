// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "data_reader.h"
#include "basic_cpu_processor.h"
#include "ctime"


bool LineItemFilter(LineItem item) {
	return item.order_key == 1;
}


int _tmain(int argc, TCHAR* argv[]) {
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	double duration;
	std::clock_t start = std::clock();

	std::vector<LineItem> items = ReadAllLineItems("D:\\school\\experimentation-project\\tpch_2_17_0\\dbgen\\Debug\\lineitem.tbl");
	std::cout << "done reading\n";

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Reading took " << duration << "ms\n";

	BasicCPUProcessor<LineItem> processor(items);
	start = std::clock();
	std::cout << "Number of elements: " << processor.Filter(&LineItemFilter).size() << "\n";

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "Filtering took " << duration << "ms\n";

	std::cin.get();
	return 0;
}