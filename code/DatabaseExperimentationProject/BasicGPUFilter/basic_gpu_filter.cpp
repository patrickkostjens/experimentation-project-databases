#include "stdafx.h"
#include "basic_gpu_filter.h"
#include "vector"

extern void filter(std::vector<LineItem> items);

void gpu_filter(std::vector<LineItem> items)
{
	filter(items);
}
