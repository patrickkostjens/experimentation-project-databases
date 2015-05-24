#include "stdafx.h"
#include "iostream"
#include "boost\program_options.hpp"
#include "command_line_options.h"

namespace po = boost::program_options;

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