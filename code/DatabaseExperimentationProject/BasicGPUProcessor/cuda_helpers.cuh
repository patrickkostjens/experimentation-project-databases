#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdafx.h"
#include <ctime>
#include <stdio.h>

double GetElapsedTime(clock_t& since);

void handleCudaError(cudaError_t status);

template<typename TItem>
__global__ void filterKernel(TItem *d_item, bool *d_result, size_t d_totalCount);
