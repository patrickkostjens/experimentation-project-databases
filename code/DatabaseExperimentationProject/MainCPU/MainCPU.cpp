// MainCPU.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"
#include "cpu_query_handler.h"
#include "ctime"

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

int _tmain(const int argc, const TCHAR* argv[]) {
	std::clock_t total_start = std::clock();
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";

	double duration;
	std::clock_t start = std::clock();

	ExecuteCPUQuery(options.query);

	double total_duration = GetElapsedTime(total_start);
	std::cout << "Total took " << total_duration << "ms\n";

	std::cin.get();
	return 0;
}
