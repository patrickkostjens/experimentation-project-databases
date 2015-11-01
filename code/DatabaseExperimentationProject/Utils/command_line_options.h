#ifndef COMMAND_LINE_OPTIONS_H
#define COMMAND_LINE_OPTIONS_H
enum ProcessingMode {
	CPU,
	GPU,
	ALL,
	NUM_MODES
};

enum Query {
	SIMPLE_LINE_ITEM,
	SIMPLE_ORDERS,
	SORT_MERGE_JOIN_LINE_ITEM_ORDERS,
	NUM_QUERIES
};

struct CommandLineOptions {
	Query query;
	ProcessingMode processing_mode;
};
#endif

const CommandLineOptions GetCommandLineOptions(const int& argc, const TCHAR* argv[]);
