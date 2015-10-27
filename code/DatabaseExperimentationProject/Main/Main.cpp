// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "cpu_query_handler.h"
#include "gpu_query_handler.h"
#include "helpers.h"
#include "performance_metrics.h"


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

	// This should help detect memory leaks
	std::cout << "Final memory usage: ";
	PrintMemoryUsage();

	std::cin.get();
	return 0;
}
