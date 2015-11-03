#include "stdafx.h"
#include "unified_memory_gpu_filter.h"

template<typename TItem>
extern std::vector<TItem>& filter_um(std::vector<TItem>& items);

template<typename TItem>
std::vector<TItem>& um_gpu_filter(std::vector<TItem>& items)
{
	return filter_um(items);
}
