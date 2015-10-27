#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "models.h"
#include <stdio.h>
#include "iostream"
#include "vector"
#include "ctime"

template<typename TItem>
__global__ void filterKernel(TItem *item, bool *result) {
	//This should never be called, but exceptions are not supported; only specialized implementations allowed
}

template<> __global__ void filterKernel<LineItem>(LineItem *item, bool *result) {
	int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	
	result[threadIndex] = item[threadIndex].order_key == 1;
}

template<> __global__ void filterKernel<Order>(Order *item, bool *result) {
	int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	result[threadIndex] = item[threadIndex].order_status == 'O';
}

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

void handleCudaError(cudaError_t status) {
	if (status != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s", cudaGetErrorString(status));
		throw cudaGetErrorString(status);
	}
}

template<typename TItem>
std::vector<TItem>& filter(std::vector<TItem>& items) {
	std::clock_t start = std::clock();
	std::vector<TItem>& returnValue = *new std::vector<TItem>();
	int count = items.size();

	TItem *deviceItems;
	bool *deviceResults = false;
	bool *hostResults = false;
	hostResults = (bool*)malloc(count * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));
	// Reserve room for input in GPU memory
	handleCudaError(cudaMalloc((void**)&deviceItems, count * sizeof(TItem)));
	// Copy input to GPU
	handleCudaError(cudaMemcpy(deviceItems, &items[0], count * sizeof(TItem), cudaMemcpyHostToDevice));
	// Reserve room for results in GPU memory
	handleCudaError(cudaMalloc((void**)&deviceResults, count * sizeof(bool)));

	std::cout << "GPU allocation and copying took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	const int threadsPerBlock = 1024;
	int blocks = (int)ceil((float)count / threadsPerBlock);
	filterKernel<TItem> <<<blocks, threadsPerBlock>>>(deviceItems, deviceResults);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "GPU filtering took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	// Copy output vector from GPU buffer to host memory.
	handleCudaError(cudaMemcpy(hostResults, deviceResults, count * sizeof(bool), cudaMemcpyDeviceToHost));

	for (int i = 0; i < count; i++)	{
		if (hostResults[i]) {
			returnValue.push_back(items[i]);
		}
	}

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(start) << "ms\n";

	// Cleanup
	free(hostResults);
	cudaDeviceReset();

	return returnValue;
}

template std::vector<LineItem>& filter<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter<Order>(std::vector<Order>& items);
