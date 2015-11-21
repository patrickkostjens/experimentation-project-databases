#include "stdafx.h"
#include "cuda_helpers.cuh"

double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

void handleCudaError(cudaError_t status) {
	if (status != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s", cudaGetErrorString(status));
		cudaDeviceReset();
		throw cudaGetErrorString(status);
	}
}
