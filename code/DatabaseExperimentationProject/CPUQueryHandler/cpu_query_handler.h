#include "../Utils/command_line_options.h"

#ifdef CPUQUERYHANDLER_EXPORTS
#define CPUQUERYHANDLER_API __declspec(dllexport)
#else
#define CPUQUERYHANDLER_API __declspec(dllimport)
#endif

CPUQUERYHANDLER_API void ExecuteCPUQuery(Query query);
