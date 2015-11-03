#include "stdafx.h"
#include "models.h"

#ifdef BASICGPUPROCESSOR_EXPORTS
#define BASICGPUPROCESSOR_API __declspec(dllexport) 
#else
#define BASICGPUPROCESSOR_API __declspec(dllimport) 
#endif

template<typename TItem>
BASICGPUPROCESSOR_API std::vector<TItem>& um_gpu_filter(std::vector<TItem>& items);

template BASICGPUPROCESSOR_API std::vector<LineItem>& um_gpu_filter<LineItem>(std::vector<LineItem>& items);
template BASICGPUPROCESSOR_API std::vector<Order>& um_gpu_filter<Order>(std::vector<Order>& items);
