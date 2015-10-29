#include <algorithm>

template<typename T, typename Comparable>
void sort_with_lambda_selector(std::vector<T> &items, Comparable(*selector)(const T)) {
	std::sort(items.begin(), items.end(), [&](const T & a, const T & b) -> bool {
		return selector(a) < selector(b);
	});
};
