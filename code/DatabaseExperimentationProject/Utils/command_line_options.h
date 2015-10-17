enum ProcessingMode {
	CPU,
	GPU,
	ALL,
	NUM_MODES
};

struct CommandLineOptions {
	int query;
	ProcessingMode processing_mode;
};

const CommandLineOptions& GetCommandLineOptions(const int& argc, const TCHAR* argv[]);
