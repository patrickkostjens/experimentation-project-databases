// MainCPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <ctime>
#include "../Utils/command_line_options.h"
#include "../CPUQueryHandler/cpu_query_handler.h"
#include "../Utils/performance_metrics.h"

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

int _tmain(const int argc, const TCHAR* argv[]) {
	std::clock_t start = std::clock();
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";

	ExecuteCPUQuery(options.query);

	std::cout << "Total took " << GetElapsedTime(start) << "ms\n";

	// This should help detect memory leaks
	std::cout << "Final memory usage: ";
	PrintMemoryUsage();

	std::cin.get();
	return 0;
}
