enum ProcessingMode {
	CPU,
	NUM_MODES
};

struct CommandLineOptions {
	int query;
	ProcessingMode processing_mode;
};

CommandLineOptions GetCommandLineOptions(int argc, TCHAR* argv[]);
