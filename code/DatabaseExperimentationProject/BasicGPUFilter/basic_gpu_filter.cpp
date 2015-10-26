#include "stdafx.h"
#include "basic_gpu_filter.h"
#include "vector"

template<typename TItem>
extern std::vector<TItem>& filter(std::vector<TItem>& items);

template<typename TItem>
std::vector<TItem>& gpu_filter(std::vector<TItem>& items)
{
	return filter(items);
}
