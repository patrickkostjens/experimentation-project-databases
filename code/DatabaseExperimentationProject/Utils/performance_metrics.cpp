#include "stdafx.h"
#include "windows.h"
#include "psapi.h"

void PrintMemoryUsage()
{
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));

	const float MBConversion = 1024 * 1024;
	printf("%6.2fMB\n", (float)pmc.WorkingSetSize / MBConversion);
}
