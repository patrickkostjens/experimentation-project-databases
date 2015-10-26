#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "models.h"

#include <stdio.h>
#include "iostream"
#include "vector"
#include "ctime"

__global__ void filterKernel(LineItem *item, bool *result)
{
	int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	result[threadIndex] = item[threadIndex].order_key == 1;
}

inline double GetElapsedTime(clock_t& since) {
	return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
}

std::vector<LineItem>& filter(std::vector<LineItem>& items)
{
	std::clock_t start = std::clock();

	std::vector<LineItem>& returnValue = *new std::vector<LineItem>();

	int count = items.size();

	LineItem *deviceItems;
	bool *results = false;
	bool *host_results = false;
	host_results = (bool*)malloc(count * sizeof(bool));

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return returnValue;
	}

	// Reserve room for input in GPU memory
	cudaStatus = cudaMalloc((void**)&deviceItems, count * sizeof(LineItem));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return returnValue;
	}

	// Copy input to GPU
	cudaStatus = cudaMemcpy(deviceItems, &items[0], count * sizeof(LineItem), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return returnValue;
	}

	// Reserve room for results in GPU memory
	cudaStatus = cudaMalloc((void**)&results, count * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return returnValue;
	}

	double duration = GetElapsedTime(start);
	std::cout << "GPU allocation and copying took " << duration << "ms\n";

	start = std::clock();

	const int threadsPerBlock = 1024;
	int blocks = (int)ceil((float)count / threadsPerBlock);

	filterKernel<TItem> <<<blocks, threadsPerBlock>>>(deviceItems, results);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return returnValue;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return returnValue;
	}

	duration = GetElapsedTime(start);
	std::cout << "GPU filtering took " << duration << "ms\n";
	start = std::clock();

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_results, results, count * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return returnValue;
	}

	for (int i = 0; i < count; i++)	{
		if (host_results[i]) {
			returnValue.push_back(items[i]);
		}
	}

	duration = GetElapsedTime(start);
	std::cout << "GPU reconstructing results (on CPU) took " << duration << "ms\n";

	// Cleanup
	free(host_results);
	cudaFree(results);
	cudaFree(deviceItems);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed\n");
		return returnValue;
	}

	return returnValue;
}

