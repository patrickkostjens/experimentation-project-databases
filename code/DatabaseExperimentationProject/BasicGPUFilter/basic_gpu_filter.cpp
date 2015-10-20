#include "stdafx.h"
#include "basic_gpu_filter.h"
#include "vector"

extern std::vector<LineItem>& filter(std::vector<LineItem>& items);

std::vector<LineItem>& gpu_filter(std::vector<LineItem>& items)
{
	return filter(items);
}
