#include "stdafx.h"
#include "models.h"

#ifdef BASICGPUFILTER_EXPORTS
#define BASICGPUFILTER_API __declspec(dllexport) 
#else
#define BASICGPUFILTER_API __declspec(dllimport) 
#endif

template<typename TItem>
BASICGPUFILTER_API std::vector<TItem>& um_gpu_filter(std::vector<TItem>& items);

template BASICGPUFILTER_API std::vector<LineItem>& um_gpu_filter<LineItem>(std::vector<LineItem>& items);
template BASICGPUFILTER_API std::vector<Order>& um_gpu_filter<Order>(std::vector<Order>& items);
