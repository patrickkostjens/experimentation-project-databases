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

template<typename TItem>
__global__ void filterKernel(TItem *d_item, bool *d_result, size_t d_totalCount) {
	//This should never be called, but exceptions are not supported; only specialized implementations allowed
}

template<> __global__ void filterKernel<LineItem>(LineItem *d_item, bool *d_result, size_t d_totalCount) {
	size_t d_threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (d_threadIndex < d_totalCount) {
		d_result[d_threadIndex] = d_item[d_threadIndex].order_key == 1;
	}
}

template<> __global__ void filterKernel<Order>(Order *d_item, bool *d_result, size_t d_totalCount) {
	size_t d_threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (d_threadIndex < d_totalCount) {
		d_result[d_threadIndex] = d_item[d_threadIndex].order_status == 'O';
	}
}
