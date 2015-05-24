// Main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "iostream"
#include "boost\program_options.hpp"

namespace po = boost::program_options;

struct command_line_options {
	int query;
};

command_line_options get_command_line_options(int argc, _TCHAR* argv[]) {
	po::options_description desc("Options");
	desc.add_options()
		("query", po::value<int>()->required(), "[Required] Set query to run")
		("help", "Show this help message");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		std::cout << desc << "\n";
		exit(0);
	}

	command_line_options options;

	try {
		po::notify(vm);

		options.query = vm["query"].as<int>();
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
	command_line_options options;
	options = get_command_line_options(argc, argv);

	std::cout << "Query: " << options.query;

	return 0;
}

