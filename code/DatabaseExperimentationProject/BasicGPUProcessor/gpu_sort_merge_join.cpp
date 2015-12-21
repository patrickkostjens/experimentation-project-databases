#include "stdafx.h"
#include "gpu_sort_merge_join.h"

template<typename Left, typename Right>
extern std::vector<std::tuple<Left, Right>>& sort_merge_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems);

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& gpu_sort_merge_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems) {
	return sort_merge_join(leftItems, rightItems);
}
