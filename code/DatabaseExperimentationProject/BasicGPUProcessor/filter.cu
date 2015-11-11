#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <ctime>

template<typename TItem>
__global__ void filterKernel(TItem *item, bool *result, size_t totalCount) {
	//This should never be called, but exceptions are not supported; only specialized implementations allowed
}

template<> __global__ void filterKernel<LineItem>(LineItem *item, bool *result, size_t totalCount) {
	size_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (threadIndex < totalCount) {
		result[threadIndex] = item[threadIndex].order_key == 1;
	}
}

template<> __global__ void filterKernel<Order>(Order *item, bool *result, size_t totalCount) {
	size_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (threadIndex < totalCount) {
		result[threadIndex] = item[threadIndex].order_status == 'O';
	}
}

inline double GetElapsedTime(clock_t& since) {
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
std::vector<TItem>& filter_standard(std::vector<TItem>& items) {
	std::clock_t start = std::clock();
	std::vector<TItem>& returnValue = *new std::vector<TItem>();
	size_t count = items.size();

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
	filterKernel<TItem> <<<blocks, threadsPerBlock>>>(deviceItems, deviceResults, count);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "GPU filtering took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	// Copy output vector from GPU buffer to host memory.
	handleCudaError(cudaMemcpy(hostResults, deviceResults, count * sizeof(bool), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < count; i++)	{
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

template std::vector<LineItem>& filter_standard<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter_standard<Order>(std::vector<Order>& items);

template<typename TItem>
std::vector<TItem>& filter_um(std::vector<TItem>& items) {
	std::clock_t start = std::clock();
	std::vector<TItem>& returnValue = *new std::vector<TItem>();
	size_t count = items.size();

	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));

	TItem *managedItems;
	// Reserve room for input items in unified memory
	handleCudaError(cudaMallocManaged(&managedItems, count * sizeof(TItem)));

	memcpy(managedItems, &items[0], count * sizeof(TItem));

	bool *managedResults = false;

	// Reserve room for results in unified memory
	handleCudaError(cudaMallocManaged(&managedResults, count * sizeof(bool)));

	std::cout << "GPU managed allocation and copying took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	const int threadsPerBlock = 1024;
	int blocks = (int)ceil((float)count / threadsPerBlock);
	filterKernel<TItem> << <blocks, threadsPerBlock >> >(managedItems, managedResults, count);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "GPU filtering took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	for (unsigned int i = 0; i < count; i++) {
		if (managedResults[i]) {
			returnValue.push_back(items[i]);
		}
	}

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(start) << "ms\n";

	// Cleanup
	cudaFree(managedItems);
	cudaFree(managedResults);
	cudaDeviceReset();

	return returnValue;
}

template std::vector<LineItem>& filter_um<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter_um<Order>(std::vector<Order>& items);

template<typename TItem>
std::vector<TItem>& filter_async(std::vector<TItem>& items) {
	std::clock_t start = std::clock();
	std::vector<TItem>& returnValue = *new std::vector<TItem>();
	size_t count = items.size();

	TItem *deviceItems;
	TItem *pinnedItems;
	bool *deviceResults = false;
	bool *hostResults = false;
	hostResults = (bool*)malloc(count * sizeof(bool));
	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));

	// Prepare streams
	const int streamCount = 4;
	cudaStream_t *streams;
	streams = (cudaStream_t*)malloc(streamCount * sizeof(cudaStream_t));
	for (int i = 0; i < streamCount; i++) {
		handleCudaError(cudaStreamCreate(&streams[i]));
	}

	// Reserve pinned host memory for data
	handleCudaError(cudaMallocHost((void**)&pinnedItems, count * sizeof(TItem)));
	// Copy input to pinned memory
	memcpy(pinnedItems, &items[0], count * sizeof(TItem));

	// Reserve room for input in GPU memory
	handleCudaError(cudaMalloc((void**)&deviceItems, count * sizeof(TItem)));
	// Reserve room for results in GPU memory
	handleCudaError(cudaMalloc((void**)&deviceResults, count * sizeof(bool)));

	size_t perStreamCount = count / streamCount;
	const int threadsPerBlock = 1024;
	int blocks = (int)ceil((float)count / threadsPerBlock);

	/* Depending on the GPU's capabilities this way of calling or calling all three CUDA functions in a single loop might be faster.
	   For details, see: http://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/ */
	// Copy input to GPU
	for (int i = 0; i < streamCount; i++) {
		size_t transferCount = perStreamCount;
		if (i == streamCount - 1) transferCount = count - i * transferCount;

		handleCudaError(cudaMemcpyAsync(&deviceItems[i*perStreamCount], &pinnedItems[i*perStreamCount], transferCount * sizeof(TItem), cudaMemcpyHostToDevice, streams[i]));
	}
	// Execute kernels
	for (int i = 0; i < streamCount; i++) {
		size_t transferCount = perStreamCount;
		if (i == streamCount - 1) transferCount = count - i * transferCount;

		filterKernel<TItem> << <blocks, threadsPerBlock, 0, streams[i] >> >(&deviceItems[i*perStreamCount], &deviceResults[i*perStreamCount], transferCount);
	}
	// Copy output vector from GPU buffer to host memory.
	for (int i = 0; i < streamCount; i++) {
		size_t transferCount = perStreamCount;
		if (i == streamCount - 1) transferCount = count - i * transferCount;

		handleCudaError(cudaMemcpyAsync(&hostResults[i*perStreamCount], &deviceResults[i*perStreamCount], transferCount * sizeof(bool), cudaMemcpyDeviceToHost, streams[i]));
	}

	// cudaDeviceSynchronize waits for the kernel and copy operations to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	for (int i = 0; i < streamCount; i++) {
		handleCudaError(cudaStreamDestroy(streams[i]));
	}

	std::cout << "GPU allocation, copying and filtering took " << GetElapsedTime(start) << "ms\n";
	start = std::clock();

	for (unsigned int i = 0; i < count; i++) {
		if (hostResults[i]) {
			returnValue.push_back(items[i]);
		}
	}

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(start) << "ms\n";

	// Cleanup
	free(hostResults);
	cudaFreeHost(pinnedItems);
	cudaDeviceReset();

	return returnValue;
}

template std::vector<LineItem>& filter_async<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter_async<Order>(std::vector<Order>& items);
