#include "cuda_helpers.cuh"
#include "stdafx.h"
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sort.h>


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

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems) {
	//TODO: Implement hash join
	return *new std::vector<std::tuple<Left, Right>>();
}

template std::vector<std::tuple<Order, LineItem>>& hash_join<Order, LineItem>(std::vector<Order>& orders, std::vector<LineItem>& items);
