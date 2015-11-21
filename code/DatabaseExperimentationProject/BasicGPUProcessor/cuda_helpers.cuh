#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>

double GetElapsedTime(clock_t& since);

void handleCudaError(cudaError_t status);
