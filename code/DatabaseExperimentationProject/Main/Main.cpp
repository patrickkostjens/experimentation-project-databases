// Main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "boost\program_options.hpp"

namespace po = boost::program_options;

enum ProcessingMode {
	CPU,
	NUM_MODES
};

struct CommandLineOptions {
	int query;
	ProcessingMode processing_mode;
};

ProcessingMode CastProcessingMode(int mode) {
	if (mode < ProcessingMode::NUM_MODES) {
		return static_cast<ProcessingMode>(mode);
	}
	else {
		throw std::exception("Invalid value for --processingMode");
	}
}

CommandLineOptions GetCommandLineOptions(int argc, _TCHAR* argv[]) {
	po::options_description desc("Options");
	desc.add_options()
		("query", po::value<int>()->required(), "[Required] Set query to run")
		("processingMode", po::value<int>()->required(), "[Required] Mode that should be used to process the query")
		("help", "Show this help message");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		exit(0);
	}

	CommandLineOptions options;

	try {
		po::notify(vm);

		options.query = vm["query"].as<int>();
		options.processing_mode = CastProcessingMode(vm["processingMode"].as<int>());
		
	}
	catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n"
			<< "Use --help for options\n";
		exit(-1);
	}
	catch (...) {
		std::cerr << "Unkown error!\n"
			<< "Use --help for options\n";
		exit(-1);
	}
	return options;
}

int _tmain(int argc, _TCHAR* argv[]) {
	CommandLineOptions options;
	options = GetCommandLineOptions(argc, argv);

	std::cout << "Query: " << options.query << "\n";
	std::cout << "ProcessingMode: " << options.processing_mode << "\n";

	return 0;
}

