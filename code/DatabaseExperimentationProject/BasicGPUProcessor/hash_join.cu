#include "cuda_helpers.cuh"
#include "stdafx.h"
#include <iostream>

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems) {
	//TODO: Implement hash join
	return *new std::vector<std::tuple<Left, Right>>();
}

template std::vector<std::tuple<Order, LineItem>>& hash_join<Order, LineItem>(std::vector<Order>& orders, std::vector<LineItem>& items);
