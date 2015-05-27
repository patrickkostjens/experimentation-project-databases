// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "command_line_options.h"


int _tmain(int argc, TCHAR* argv[]) {
	CommandLineOptions options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	return 0;
}

