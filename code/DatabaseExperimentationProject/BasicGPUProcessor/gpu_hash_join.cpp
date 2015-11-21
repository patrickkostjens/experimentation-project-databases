#include "stdafx.h"
#include "gpu_hash_join.h"

template<typename Left, typename Right>
extern std::vector<std::tuple<Left, Right>>& hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems);

template<typename Left, typename Right>
std::vector<std::tuple<Left, Right>>& gpu_hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems) {
	return hash_join(leftItems, rightItems);
}
