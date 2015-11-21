#include "stdafx.h"
#include "cuda_helpers.cuh"
#include <iostream>

template<typename TItem>
std::vector<TItem>& create_result_vector(std::vector<TItem>& h_items, bool* h_results) {
	std::vector<TItem>& h_returnValue = *new std::vector<TItem>();
	size_t h_count = h_items.size();

	for (unsigned int h_i = 0; h_i < h_count; h_i++) {
		if (h_results[h_i]) {
			h_returnValue.push_back(h_items[h_i]);
		}
	}

	return h_returnValue;
}

template<typename TItem>
std::vector<TItem>& filter_standard(std::vector<TItem>& h_items) {
	std::clock_t h_start = std::clock();
	size_t h_count = h_items.size();

	TItem *d_items;
	bool *d_results = false;
	bool *h_results = false;
	h_results = (bool*)malloc(h_count * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));
	// Reserve room for input in GPU memory
	handleCudaError(cudaMalloc((void**)&d_items, h_count * sizeof(TItem)));
	// Copy input to GPU
	handleCudaError(cudaMemcpy(d_items, &h_items[0], h_count * sizeof(TItem), cudaMemcpyHostToDevice));
	// Reserve room for results in GPU memory
	handleCudaError(cudaMalloc((void**)&d_results, h_count * sizeof(bool)));

	std::cout << "GPU allocation and copying took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	const int h_threadsPerBlock = 1024;
	int h_blocks = (int)ceil((float)h_count / h_threadsPerBlock);
	filterKernel<TItem> <<<h_blocks, h_threadsPerBlock>>>(d_items, d_results, h_count);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "GPU filtering took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Copy output vector from GPU buffer to host memory.
	handleCudaError(cudaMemcpy(h_results, d_results, h_count * sizeof(bool), cudaMemcpyDeviceToHost));

	std::vector<TItem>& h_returnValue = create_result_vector(h_items, h_results);

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(h_start) << "ms\n";

	// Cleanup
	free(h_results);
	cudaDeviceReset();

	return h_returnValue;
}

template std::vector<LineItem>& filter_standard<LineItem>(std::vector<LineItem>& h_items);
template std::vector<Order>& filter_standard<Order>(std::vector<Order>& h_items);

template<typename TItem>
std::vector<TItem>& filter_um(std::vector<TItem>& h_items) {
	std::clock_t h_start = std::clock();
	size_t h_count = h_items.size();

	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));

	TItem *m_items;
	// Reserve room for input items in unified memory
	handleCudaError(cudaMallocManaged(&m_items, h_count * sizeof(TItem)));

	memcpy(m_items, &h_items[0], h_count * sizeof(TItem));

	bool *m_results = false;

	// Reserve room for results in unified memory
	handleCudaError(cudaMallocManaged(&m_results, h_count * sizeof(bool)));

	std::cout << "GPU managed allocation and copying took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	const int h_threadsPerBlock = 1024;
	int h_blocks = (int)ceil((float)h_count / h_threadsPerBlock);
	filterKernel<TItem> <<<h_blocks, h_threadsPerBlock>>>(m_items, m_results, h_count);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "GPU filtering took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	std::vector<TItem>& h_returnValue = create_result_vector(h_items, m_results);

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(h_start) << "ms\n";

	// Cleanup
	cudaFree(m_items);
	cudaFree(m_results);
	cudaDeviceReset();

	return h_returnValue;
}

template std::vector<LineItem>& filter_um<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter_um<Order>(std::vector<Order>& items);

size_t get_transfer_count(int h_index, size_t h_perStreamCount, size_t h_totalCount, const int h_streamCount) {
	size_t h_transferCount = h_perStreamCount;
	if (h_index == h_streamCount - 1) h_transferCount = h_totalCount - h_index * h_transferCount;
	return h_transferCount;
}

template<typename TItem>
std::vector<TItem>& filter_async(std::vector<TItem>& h_items) {
	std::clock_t h_start = std::clock();
	size_t h_count = h_items.size();

	TItem *d_items;
	TItem *h_pinnedItems;
	bool *d_results = false;
	bool *h_results = false;
	h_results = (bool*)malloc(h_count * sizeof(bool));
	// Choose which GPU to run on, change this on a multi-GPU system.
	handleCudaError(cudaSetDevice(0));

	// Prepare streams
	const int h_streamCount = 4;
	cudaStream_t *h_streams;
	h_streams = (cudaStream_t*)malloc(h_streamCount * sizeof(cudaStream_t));
	for (int h_i = 0; h_i < h_streamCount; h_i++) {
		handleCudaError(cudaStreamCreate(&h_streams[h_i]));
	}

	// Reserve pinned host memory for data
	handleCudaError(cudaMallocHost((void**)&h_pinnedItems, h_count * sizeof(TItem)));
	// Copy input to pinned memory
	memcpy(h_pinnedItems, &h_items[0], h_count * sizeof(TItem));

	// Reserve room for input in GPU memory
	handleCudaError(cudaMalloc((void**)&d_items, h_count * sizeof(TItem)));
	// Reserve room for results in GPU memory
	handleCudaError(cudaMalloc((void**)&d_results, h_count * sizeof(bool)));

	size_t h_perStreamCount = h_count / h_streamCount;
	const int h_threadsPerBlock = 1024;
	int h_blocks = (int)ceil((float)h_count / h_threadsPerBlock);

	/* Depending on the GPU's capabilities this way of calling or calling all three CUDA functions in a single loop might be faster.
	   For details, see: http://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/ */
	// Copy input to GPU
	for (int h_i = 0; h_i < h_streamCount; h_i++) {
		size_t h_transferCount = get_transfer_count(h_i, h_perStreamCount, h_count, h_streamCount);

		handleCudaError(cudaMemcpyAsync(&d_items[h_i * h_perStreamCount], 
			&h_pinnedItems[h_i * h_perStreamCount],
			h_transferCount * sizeof(TItem), 
			cudaMemcpyHostToDevice, 
			h_streams[h_i]));
	}
	// Execute kernels
	for (int h_i = 0; h_i < h_streamCount; h_i++) {
		size_t h_transferCount = get_transfer_count(h_i, h_perStreamCount, h_count, h_streamCount);

		filterKernel<TItem> <<<h_blocks, h_threadsPerBlock, 0, h_streams[h_i]>>>(&d_items[h_i * h_perStreamCount], &d_results[h_i * h_perStreamCount], h_transferCount);
	}
	// Copy output vector from GPU buffer to host memory.
	for (int h_i = 0; h_i < h_streamCount; h_i++) {
		size_t h_transferCount = get_transfer_count(h_i, h_perStreamCount, h_count, h_streamCount);

		handleCudaError(cudaMemcpyAsync(&h_results[h_i*h_perStreamCount], &d_results[h_i*h_perStreamCount], h_transferCount * sizeof(bool), cudaMemcpyDeviceToHost, h_streams[h_i]));
	}

	// cudaDeviceSynchronize waits for the kernel and copy operations to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	for (int h_i = 0; h_i < h_streamCount; h_i++) {
		handleCudaError(cudaStreamDestroy(h_streams[h_i]));
	}

	std::cout << "GPU allocation, copying and filtering took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	std::vector<TItem>& h_returnValue = create_result_vector(h_items, h_results);

	std::cout << "GPU reconstructing results (on CPU) took " << GetElapsedTime(h_start) << "ms\n";

	// Cleanup
	free(h_results);
	cudaFreeHost(h_pinnedItems);
	cudaDeviceReset();

	return h_returnValue;
}

template std::vector<LineItem>& filter_async<LineItem>(std::vector<LineItem>& items);
template std::vector<Order>& filter_async<Order>(std::vector<Order>& items);
