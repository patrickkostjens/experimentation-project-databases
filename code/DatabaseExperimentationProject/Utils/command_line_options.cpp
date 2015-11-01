// Defines and parses the command line options for the console application.

#include "stdafx.h"
#include <iostream>
#include "command_line_options.h"

#ifndef UNICODE
typedef std::string String;
#else
typedef std::wstring String;
#endif

const ProcessingMode CastProcessingMode(const int& mode) {
	if (mode < ProcessingMode::NUM_MODES) {
		return static_cast<ProcessingMode>(mode);
	}
	else {
		std::cerr << "Invalid value for \"--processingMode\"\n";
		exit(-1);
	}
}

const Query CastQuery(const int& query) {
	if (query < Query::NUM_QUERIES) {
		return static_cast<Query>(query);
	}
	else {
		std::cerr << "Invalid value for \"--query\"\n";
		exit(-1);
	}
}

const TCHAR* GetCmdOption(const TCHAR* args[], const int& count, const std::string & option)
{
	String woption(option.begin(), option.end());
	for (int i = 0; i < count; i++) {
		String s = args[i];
		if (s == woption && i + 1 < count) {
			return args[i + 1];
		}
	}
	return 0;
}

const bool CmdOptionExists(const TCHAR* args[], const int& count, const std::string& option)
{
	String woption(option.begin(), option.end());
	for (int i = 0; i < count; i++) {
		String s = args[i];
		if (s == woption) {
			return true;
		}
	}
	return false;
}

void PrintHelp() {
	std::cout << "Required options:\n"
		<< "\t--query\t\t\tSet query to run\n"
		<< "\t--processingMode\tMode that should be used to process the query\n"
		<< "\n"
		<< "Other options:\n"
		<< "\t--help\t\t\tShow this help message\n";
}

void ErrorMissingArgument(const char* argument) {
	std::cerr << "Required argument \"" << argument << "\" missing. \n" << "Use \"--help\" for options.\n";
}

const CommandLineOptions GetCommandLineOptions(const int& argc, const TCHAR* argv[])
{
	if (CmdOptionExists(argv, argc, "--help"))
	{
		PrintHelp();
		exit(0);
	}

	const TCHAR* query = GetCmdOption(argv, argc, "--query");
	if (!query) {
		ErrorMissingArgument("--query");
		exit(-1);
	}

	const TCHAR* processing_mode = GetCmdOption(argv, argc, "--processingMode");
	if (!processing_mode) {
		ErrorMissingArgument("--processingMode");
		exit(-1);
	}

	CommandLineOptions options;
	options.query = CastQuery(_ttoi(query));
	options.processing_mode = CastProcessingMode(_ttoi(processing_mode));

	return options;
}
