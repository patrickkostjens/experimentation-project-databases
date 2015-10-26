#include "stdafx.h"
#include<models.h>
#include "vector"

#ifdef BASICGPUFILTER_EXPORTS
#define BASICGPUFILTER_API __declspec(dllexport) 
#else
#define BASICGPUFILTER_API __declspec(dllimport) 
#endif

template<typename TItem> 
BASICGPUFILTER_API std::vector<TItem>& gpu_filter(std::vector<TItem>& items);

template BASICGPUFILTER_API std::vector<LineItem>& gpu_filter<LineItem>(std::vector<LineItem>& items);
template BASICGPUFILTER_API std::vector<Order>& gpu_filter<Order>(std::vector<Order>& items);
