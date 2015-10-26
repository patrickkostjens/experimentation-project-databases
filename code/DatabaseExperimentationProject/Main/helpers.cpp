#include "stdafx.h"

double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}
