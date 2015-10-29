#include <tuple>
#include <iostream>
#include <ctime>
#include "sort_helper.h"

template <class Left, class Right>
class CPUSortMergeJoin {
private:
	std::vector<Left> _leftData;
	std::vector<Right> _rightData;

	inline double GetElapsedTime(clock_t& since) {
		return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
	}
public:
	CPUSortMergeJoin(const std::vector<Left>& leftData, const std::vector<Right>& rightData){
		_leftData = leftData;
		_rightData = rightData;
		return;
	};

	template <typename Comparable>
	std::vector<std::tuple<Left, Right>>& Join(Comparable(*leftSelector)(const Left), Comparable(*rightSelector)(const Right)){
		typedef std::tuple<Left, Right> JoinTuple;

		std::vector<JoinTuple> *result = new std::vector<JoinTuple>();

		std::clock_t start = std::clock();
		sort_with_lambda_selector<Left, Comparable>(_leftData, leftSelector);
		sort_with_lambda_selector<Right, Comparable>(_rightData, rightSelector);
		std::cout << "Sorting took " << GetElapsedTime(start) << "ms\n";

		start = std::clock();
		std::vector<Left>::iterator leftIterator = _leftData.begin();
		std::vector<Right>::iterator rightIterator = _rightData.begin();

		while (leftIterator != _leftData.end() && rightIterator != _rightData.end()) {
			if (leftSelector(*leftIterator) == rightSelector(*rightIterator)) {
				std::vector<Right>::iterator subIterator = rightIterator;
				while (subIterator != _rightData.end() && leftSelector(*leftIterator) == rightSelector(*subIterator)) {
					result->push_back(JoinTuple(*leftIterator, *subIterator));
					subIterator++;
				}
				leftIterator++;
			}
			else if (leftSelector(*leftIterator) < rightSelector(*rightIterator)) {
				leftIterator++;
			}
			else {
				rightIterator++;
			}
		}
		std::cout << "Merging took " << GetElapsedTime(start) << "ms\n";

		/*for (unsigned int i = 0; i < result->size(); i++) {
			JoinTuple &resultElement = result->at(i);
			std::cout << "Left: " << std::get<0>(resultElement) << "; Right: " << std::get<1>(resultElement) << "\n";
		}*/

		return *result;
	};
};
