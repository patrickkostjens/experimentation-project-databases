#include "cuda_helpers.cuh"
#include "stdafx.h"
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <algorithm>


#define MARK_VAL -1

template<typename Left, typename Right>
__global__ void join_partitions(const Left *left, const int *leftStartIndexes, const int *leftPartitionSizes,
	const Right *right, const int *rightStartIndexes, const int *rightPartitionSizes,
	thrust::tuple<Left, Right> *result, size_t partitionCount, const int *resultStartIndexes)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < partitionCount)
	{
		int leftOffset = leftStartIndexes[idx];
		int rightOffset = rightStartIndexes[idx];
		int offset = resultStartIndexes[idx];
		for (int leftIndex = 0; leftIndex < leftPartitionSizes[idx]; leftIndex++) {
			for (int rightIndex = 0; rightIndex < rightPartitionSizes[idx]; rightIndex++) {
				result[offset] = thrust::make_tuple(left[leftOffset + leftIndex], right[rightOffset + rightIndex]);
				offset++;
			}
		}
	}
}

template<typename Input>
struct order_key_selector : public thrust::unary_function<Input, int>
{
	__host__ __device__
		int operator()(const Input& input) const
	{
		return input.order_key;
	}
};

struct mark_multiply_func
{
	template <typename T1, typename T2>
	__host__ __device__
		int operator()(T1 &z1, T2 &z2){
		int res = MARK_VAL;
		if (thrust::get<0>(z1) == thrust::get<0>(z2)){
			res = thrust::get<1>(z1) * thrust::get<1>(z2);
		}
		return res;
	}
};

struct mark_test_func
{
	template <typename T>
	__host__ __device__
		bool operator()(T t){
		if (thrust::get<1>(t) == MARK_VAL) return true;
		return false;
	}
};

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& sort_merge_join(std::vector<Left>& h_leftItems, std::vector<Right>& h_rightItems) {
	std::clock_t h_start = std::clock();

	// Copy host data to the device
	thrust::device_vector<Left> d_leftItems(h_leftItems);
	thrust::device_vector<Right> d_rightItems(h_rightItems);

	// Allocate space for the row keys on the device
	thrust::device_vector<int> d_leftKeys(h_leftItems.size());
	thrust::device_vector<int> d_rightKeys(h_rightItems.size());

	std::cout << "Copying input and allocating space took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Create device vectors containing the keys for the join operation
	order_key_selector<Left> leftOperator;
	order_key_selector<Right> rightOperator;
	thrust::transform(d_leftItems.begin(), d_leftItems.end(), d_leftKeys.begin(), leftOperator);
	thrust::transform(d_rightItems.begin(), d_rightItems.end(), d_rightKeys.begin(), rightOperator);

	std::cout << "Selecting join keys took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Sort the data using the keys (used for partitioning the data)
	thrust::sort_by_key(d_leftKeys.begin(), d_leftKeys.end(), d_leftItems.begin());
	thrust::sort_by_key(d_rightKeys.begin(), d_rightKeys.end(), d_rightItems.begin());

	std::cout << "Sorting data took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Allocate space for the parition keys and sizes
	thrust::device_vector<int> d_leftCountKeys(h_leftItems.size());
	thrust::device_vector<int> d_rightCountKeys(h_rightItems.size());
	thrust::device_vector<int> d_leftCounts(h_leftItems.size());
	thrust::device_vector<int> d_rightCounts(h_rightItems.size());

	std::cout << "Allocating space for partition keys and values took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Calculate the partition keys and sizes
	auto h_newLeftEnd = thrust::reduce_by_key(d_leftKeys.begin(), d_leftKeys.end(), 
		thrust::make_constant_iterator(1), d_leftCountKeys.begin(), d_leftCounts.begin());
	auto h_newRightEnd = thrust::reduce_by_key(d_rightKeys.begin(), d_rightKeys.end(), 
		thrust::make_constant_iterator(1), d_rightCountKeys.begin(), d_rightCounts.begin());

	std::cout << "Calculating partition keys and sizes took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	int64_t h_leftCount = h_newLeftEnd.first - d_leftCountKeys.begin();
	int64_t h_rightCount = h_newRightEnd.first - d_rightCountKeys.begin();

	// Calculate partition start indexes
	// Based on http://stackoverflow.com/a/34371396/2041231
	thrust::device_vector<int> d_mergedKeys(h_leftCount + h_rightCount);
	thrust::device_vector<int> d_mergedValues(h_leftCount + h_rightCount);
	thrust::device_vector<int> d_startIndexes(h_leftCount + h_rightCount - 1);

	// Create list with keys and values for both the left and right side
	thrust::merge_by_key(d_leftCountKeys.begin(), d_leftCountKeys.begin() + h_leftCount,
		d_rightCountKeys.begin(), d_rightCountKeys.begin() + h_rightCount,
		d_leftCounts.begin(), d_rightCounts.begin(), d_mergedKeys.begin(), d_mergedValues.begin());

	// Compute multiplications of each pair of elements for which the key matches (=partition sizes)
	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.begin(), d_mergedValues.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.end() - 1, d_mergedValues.end() - 1)),
		thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.begin() + 1, d_mergedValues.begin() + 1)),
		d_startIndexes.begin(), mark_multiply_func());

	// Remove elements for which the key does not match
	size_t h_filteredResultSize = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.begin(), d_startIndexes.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.end() - 1, d_startIndexes.end())), mark_test_func()) -
		thrust::make_zip_iterator(thrust::make_tuple(d_mergedKeys.begin(), d_startIndexes.begin()));
	d_startIndexes.resize(h_filteredResultSize);

	// Compute the prefix sum to get the start indexes from the partition sizes
	thrust::exclusive_scan(d_startIndexes.begin(), d_startIndexes.begin() + h_filteredResultSize, d_startIndexes.begin());

	std::cout << "Calculating partition start indexes took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	thrust::device_vector<int> d_leftStartIndexes(h_leftCount);
	thrust::device_vector<int> d_rightStartIndexes(h_rightCount);
	thrust::exclusive_scan(d_leftCounts.begin(), d_leftCounts.begin() + h_leftCount, d_leftStartIndexes.begin());
	thrust::exclusive_scan(d_rightCounts.begin(), d_rightCounts.begin() + h_rightCount, d_rightStartIndexes.begin());

	std::cout << "Calculating join block start indexes took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	int h_joinResultSize = 0;
	if (d_startIndexes.size() > 0) {
		h_joinResultSize = *(d_startIndexes.end() - 1) + *(d_leftCounts.end() - 1) * *(d_rightCounts.end() - 1) + 1;
	}

	thrust::device_vector<thrust::tuple<Left, Right>> d_joinResult(h_joinResultSize);

	unsigned int h_blockSize = 256;
	unsigned int h_numBlocks = ((unsigned int)d_startIndexes.size() + (h_blockSize - 1)) / h_blockSize;

	join_partitions<<<h_numBlocks, h_blockSize>>>(thrust::raw_pointer_cast(d_leftItems.data()),
		thrust::raw_pointer_cast(d_leftStartIndexes.data()),
		thrust::raw_pointer_cast(d_leftCounts.data()),
		thrust::raw_pointer_cast(d_rightItems.data()),
		thrust::raw_pointer_cast(d_rightStartIndexes.data()),
		thrust::raw_pointer_cast(d_rightCounts.data()),
		thrust::raw_pointer_cast(d_joinResult.data()),
		d_startIndexes.size(),
		thrust::raw_pointer_cast(d_startIndexes.data()));
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	handleCudaError(cudaDeviceSynchronize());

	std::cout << "Calculating join result took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	thrust::host_vector<thrust::tuple<Left, Right>> h_thrustResult = d_joinResult;
	std::vector<std::tuple<Left, Right>> &h_result = *new std::vector<std::tuple<Left, Right>>(h_thrustResult.size());

	for (size_t i = 0; i < h_thrustResult.size(); i++) {
		h_result[i] = std::make_tuple(thrust::get<0>(h_thrustResult[i]), thrust::get<1>(h_thrustResult[i]));
	}
	std::cout << "Copying results to host took " << GetElapsedTime(h_start) << "ms\n";

	return h_result;
}

template std::vector<std::tuple<Order, LineItem>>& sort_merge_join<Order, LineItem>(std::vector<Order>& orders, std::vector<LineItem>& items);
