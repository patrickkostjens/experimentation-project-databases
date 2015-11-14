#ifndef COMMAND_LINE_OPTIONS_H
#define COMMAND_LINE_OPTIONS_H
enum ProcessingMode {
	CPU,
	GPU,
	ALL,
	NUM_MODES
};

enum Query {
	FILTER_LINE_ITEM,
	FILTER_ORDERS,
	INDEXED_FILTER_LINE_ITEM,
	JOIN_LINE_ITEM_ORDERS,
	NUM_QUERIES
};

struct CommandLineOptions {
	Query query;
	ProcessingMode processing_mode;
};
#endif

const CommandLineOptions GetCommandLineOptions(const int& argc, const TCHAR* argv[]);
