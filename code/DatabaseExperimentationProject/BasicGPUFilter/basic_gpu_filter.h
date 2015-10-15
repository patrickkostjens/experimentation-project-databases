#include "stdafx.h"
#include<models.h>
#include "vector"

#ifdef BASICGPUFILTER_EXPORTS
#define BASICGPUFILTER_API __declspec(dllexport) 
#else
#define BASICGPUFILTER_API __declspec(dllimport) 
#endif

BASICGPUFILTER_API void gpu_filter(std::vector<LineItem>& items);
