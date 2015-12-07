#include "cuda_helpers.cuh"
#include "stdafx.h"
#include <iostream>

#pragma region Scatter and gather
template<typename Input, typename Output>
//TODO: Implement specific scatter operation
__device__ Output scatterInput(Input d_input) {
	return d_input;
};

template<typename Input, typename Output>
__global__ void scatterKernel(Input *d_input, Output *d_output, ptrdiff_t *d_indexes, size_t d_totalCount) {
	size_t d_threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (d_threadIndex < d_totalCount) {
		d_output[d_indexes[d_threadIndex]] = scatterInput<Input, Output>(d_input[d_threadIndex]);
	}
};

template<typename Input, typename Output>
//TODO: Implement specific gather operation
__device__ Output gatherInput(Input d_input) {
	return d_input;
};

template<typename Input, typename Output>
__global__ void gatherKernel(Input *d_input, Output *d_output, ptrdiff_t *d_indexes, size_t d_totalCount) {
	size_t d_threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (d_threadIndex < d_totalCount) {
		d_output[d_threadIndex] = gatherInput<Input, Output>(d_input[d_indexes[d_threadIndex]]);
	}
};

template<typename Input, typename Output>
std::vector<Output>& scatter_gather(std::vector<Input> h_input, std::vector<ptrdiff_t> h_indexes, bool scatter) {
	if (h_indexes.size() != h_input.size()) {
		throw "Wrong number of indexes provided";
	}

	size_t h_itemCount = h_input.size();
	Input *d_input;
	Output *d_result;
	ptrdiff_t *d_indexes;

	handleCudaError(cudaSetDevice(0));
	handleCudaError(cudaMalloc((void**)&d_input, h_itemCount * sizeof(Input)));
	handleCudaError(cudaMemcpy(d_input, &h_input[0], h_itemCount * sizeof(Input), cudaMemcpyHostToDevice));
	handleCudaError(cudaMalloc((void**)&d_indexes, h_itemCount * sizeof(ptrdiff_t)));
	handleCudaError(cudaMemcpy(d_indexes, &h_indexes[0], h_itemCount * sizeof(ptrdiff_t), cudaMemcpyHostToDevice));
	handleCudaError(cudaMalloc((void**)&d_result, h_itemCount * sizeof(Output)));

	const int h_threadsPerBlock = 1024;
	int h_blocks = (int)ceil((float)h_itemCount / h_threadsPerBlock);
	if (scatter) {
		scatterKernel<Input, Output> <<<h_blocks, h_threadsPerBlock>>>(d_input, d_result, d_indexes, h_itemCount);
	}
	else {
		gatherKernel<Input, Output> <<<h_blocks, h_threadsPerBlock>>>(d_input, d_result, d_indexes, h_itemCount);
	}

	std::vector<Output>& h_result = *new std::vector<Output>();
	h_result.resize(h_itemCount);

	handleCudaError(cudaDeviceSynchronize());

	handleCudaError(cudaMemcpy(&h_result[0], d_result, h_itemCount * sizeof(Output), cudaMemcpyDeviceToHost));

	cudaDeviceReset();

	return h_result;
};
#pragma endregion

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems) {
	//TODO: Implement hash join
	return *new std::vector<std::tuple<Left, Right>>();
}

template std::vector<std::tuple<Order, LineItem>>& hash_join<Order, LineItem>(std::vector<Order>& orders, std::vector<LineItem>& items);
