
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

void filter(std::vector<LineItem>& items)
{
	std::clock_t start = std::clock();

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
		return;
	}

	cudaStatus = cudaMalloc((void**)&deviceItems, count * sizeof(LineItem));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}
	cudaStatus = cudaMemcpy(deviceItems, &items[0], count * sizeof(LineItem), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return;
	}
	cudaStatus = cudaMalloc((void**)&results, count * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "GPU allocation and copying took " << duration << "ms\n";

	start = std::clock();

	int blocks = (int)ceil((float)count / 1024);

	filterKernel <<<blocks, count / 1024 >>>(deviceItems, results);	
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_results, results, count * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return;
	}

	int resultCount = 0;
	for (int i = 0; i < count; i++)	{
		if (host_results[i]) {
			resultCount++;
		}
	}

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000;
	std::cout << "GPU filtering and result counting took " << duration << "ms\n";

	std::cout << "GPU result count: " << resultCount << "\n";
}

