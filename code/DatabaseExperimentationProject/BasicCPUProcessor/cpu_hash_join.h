#include <tuple>
#include <iostream>
#include <ctime>
#include <unordered_map>

template <class Left, class Right>
class CPUHashJoin {
private:
	const std::vector<Left> *_leftData;
	const std::vector<Right> *_rightData;

	inline double GetElapsedTime(clock_t& since) {
		return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
	}

public:
	CPUHashJoin(const std::vector<Left>& leftData, const std::vector<Right>& rightData){
		_leftData = &leftData;
		_rightData = &rightData;
		return;
	};

	template <typename Comparable>
	std::vector<std::tuple<Left, Right>>& Join(Comparable(*leftSelector)(const Left), Comparable(*rightSelector)(const Right)){
		typedef std::tuple<Left, Right> JoinTuple;

		std::clock_t start = std::clock();
		std::unordered_multimap<Comparable, const Left&> hashTable;

		for (unsigned int i = 0; i < _leftData->size(); i++) {
			const Left& element = _leftData->at(i);
			auto pair = std::pair<Comparable, const Left&>(leftSelector(element), element);
			hashTable.insert(pair);
		}
		std::cout << "Building hash table took " << GetElapsedTime(start) << "ms\n";

		start = std::clock();
		std::vector<JoinTuple>& result = *new std::vector<JoinTuple>();

		for (unsigned int i = 0; i < _rightData->size(); i++) {
			const Right& rightElement = _rightData->at(i);
			Comparable key = rightSelector(rightElement);
			auto& localResults = hashTable.find(key);
			while (localResults != hashTable.end() && localResults->first == key) {
				result.push_back(JoinTuple(localResults->second, rightElement));
				localResults++;
			}
		}
		std::cout << "Joining using hash table took " << GetElapsedTime(start) << "ms\n";
		

		/*for (unsigned int i = 0; i < result.size(); i++) {
			JoinTuple &resultElement = result.at(i);
			std::cout << "Left: " << std::get<0>(resultElement) << "; Right: " << std::get<1>(resultElement) << "\n";
		}*/
		return result;
	};
};
