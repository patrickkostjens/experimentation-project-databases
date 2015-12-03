#include "cuda_helpers.cuh"
#include "stdafx.h"
#include <iostream>

#pragma region Scatter
template<typename Input, typename Output>
//TODO: Implement specific scatter operation
__device__ Output scatterInput(Input d_input) {
	return d_input;
};

template<typename Input, typename Output>
__global__ void scatterKernel(Input *d_input, Output *d_output, size_t d_totalCount) {
	size_t d_threadIndex = threadIdx.x + blockDim.x * blockIdx.x;

	if (d_threadIndex < d_totalCount) {
		d_output[d_threadIndex] = scatterInput<Input, Output>(d_input[d_threadIndex]);
	}
};

template<typename Input, typename Output>
std::vector<Output>& scatter(std::vector<Input> h_input) {
	size_t h_itemCount = h_input.size();
	Input *d_input;
	Output *d_result;

	handleCudaError(cudaSetDevice(0));
	handleCudaError(cudaMalloc((void**)&d_input, h_itemCount * sizeof(Input)));
	handleCudaError(cudaMemcpy(d_input, &h_input[0], h_itemCount * sizeof(Input), cudaMemcpyHostToDevice));
	handleCudaError(cudaMalloc((void**)&d_result, h_itemCount * sizeof(Output)));

	const int h_threadsPerBlock = 1024;
	int h_blocks = (int)ceil((float)h_itemCount / h_threadsPerBlock);
	scatterKernel<Input, Output> <<<h_blocks, h_threadsPerBlock>>>(d_input, d_result, h_itemCount);

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
