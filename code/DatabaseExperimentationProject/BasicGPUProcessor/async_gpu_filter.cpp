#include "stdafx.h"
#include "async_gpu_filter.h"

template<typename TItem>
extern std::vector<TItem>& filter_async(std::vector<TItem>& items);

template<typename TItem>
std::vector<TItem>& gpu_filter_async(std::vector<TItem>& items)
{
	return filter_async(items);
}
