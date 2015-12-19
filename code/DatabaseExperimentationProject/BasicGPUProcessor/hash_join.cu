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
#include <algorithm>


std::vector<int>& scan(std::vector<int> h_input) {
	std::vector<int>& h_result = *new std::vector<int>();
	h_result.resize(h_input.size());
	thrust::inclusive_scan(h_input.begin(), h_input.end(), h_result.begin());
	return h_result;
}

template<typename Data>
std::vector<Data>& scatter(std::vector<Data> h_input, std::vector<int> h_indexes) {
	thrust::device_vector<Data> d_input = h_input;
	thrust::device_vector<int> d_indexes = h_indexes;
	thrust::device_vector<Data> d_result(h_input.size());
	thrust::scatter(d_input.begin(), d_input.end(), d_indexes.begin(), d_result.begin());

	std::vector<Data>& h_result = *new std::vector<Data>(h_input.size());
	thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
	return h_result;
}

template<typename Data>
std::vector<Data>& gather(std::vector<Data> h_input, std::vector<int> h_indexes) {
	thrust::device_vector<Data> d_input = h_input;
	thrust::device_vector<int> d_indexes = h_indexes;
	thrust::device_vector<Data> d_result(h_input.size());
	thrust::gather(d_input.begin(), d_input.end(), d_indexes.begin(), d_result.begin());

	std::vector<Data>& h_result = *new std::vector<Data>(h_input.size());
	thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
	return h_result;
}

// Can be used as split/partition operator by making the keys partition numbers
template<typename Data>
std::vector<Data>& sort_by_key(std::vector<Data>& h_input, std::vector<int>& h_keys) {
	thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_input.begin());
	return h_input;
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

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& hash_join(std::vector<Left>& h_leftItems, std::vector<Right>& h_rightItems) {
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

	int leftCount = h_newLeftEnd.first - d_leftCountKeys.begin();
	int rightCount = h_newRightEnd.first - d_rightCountKeys.begin();

	//TODO: Move next blocks to GPU code
	// Copy the partition keys and sizes to the host
	std::vector<int> h_leftCountKeys(leftCount);
	thrust::copy(d_leftCountKeys.begin(), h_newLeftEnd.first, h_leftCountKeys.begin());
	std::vector<int> h_rightCountKeys(rightCount);
	thrust::copy(d_rightCountKeys.begin(), h_newRightEnd.first, h_rightCountKeys.begin());
	std::vector<int> h_leftCounts(leftCount);
	thrust::copy(d_leftCounts.begin(), d_leftCounts.begin() + leftCount, h_leftCounts.begin());
	std::vector<int> h_rightCounts(rightCount);
	thrust::copy(d_rightCounts.begin(), d_rightCounts.begin() + rightCount, h_rightCounts.begin());

	std::cout << "Copying partition keys and sizes to host took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	// Calculate the start indexes for each partition
	int h_leftIndex = 0, h_rightIndex = 0;
	std::vector<thrust::tuple<int, int>> h_startIndexes;
	while (h_leftIndex < h_leftCountKeys.size() && h_rightIndex < h_rightCountKeys.size()) {
		if (h_leftCountKeys[h_leftIndex] < h_rightCountKeys[h_rightIndex]) {
			h_leftIndex++;
		}
		if (h_leftCountKeys[h_leftIndex] > h_rightCountKeys[h_rightIndex]) {
			h_rightIndex++;
		}
		int offset = 0;
		if (h_startIndexes.size() > 0) {
			offset += thrust::get<1>(h_startIndexes[h_startIndexes.size() - 1]);
		}
		h_startIndexes.push_back(thrust::make_tuple(h_leftCountKeys[h_leftIndex], offset + h_leftCounts[h_leftIndex] * h_rightCounts[h_rightIndex]));
		h_leftIndex++;
		h_rightIndex++;
	}
	thrust::device_vector<thrust::tuple<int, int>> d_startIndexes(h_startIndexes);
	std::cout << "Calculating partition start indexes took " << GetElapsedTime(h_start) << "ms\n";
	h_start = std::clock();

	return *new std::vector<std::tuple<Left, Right>>();
}

template std::vector<std::tuple<Order, LineItem>>& hash_join<Order, LineItem>(std::vector<Order>& orders, std::vector<LineItem>& items);
