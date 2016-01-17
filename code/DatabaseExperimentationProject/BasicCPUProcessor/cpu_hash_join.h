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

	std::tuple<Left, Right> MakeTuple(const Left& left, const Right& right) {
		return std::tuple<Left, Right>(left, right);
	}

	std::tuple<Left, Right> MakeTuple(const Right& right, const Left& left) {
		return std::tuple<Left, Right>(left, right);
	}

	template <typename Key, class Value>
	void CreateHashTable(const std::vector<Value>* input, Key(*keySelector)(const Value), 
		std::unordered_multimap<Key, const Value&>& hashTable) {
		std::clock_t start = std::clock();

		// Add every element to the hash table
		for (unsigned int i = 0; i < input->size(); i++) {
			const Value& element = input->at(i);
			auto pair = std::pair<Key, const Value&>(keySelector(element), element);
			hashTable.insert(pair);
		}
#if DEBUG
		std::cout << "Building hash table took " << GetElapsedTime(start) << "ms\n";
#endif
	}

	template <typename Comparable, typename Hashed, typename Scanned>
	std::vector<std::tuple<Left, Right>>& PerformJoin(std::unordered_multimap<Comparable, const Hashed&> hashTable,
		const std::vector<Scanned> *otherInput, Comparable(*otherSelector)(const Scanned)) {
		std::clock_t start = std::clock();
		std::vector<std::tuple<Left, Right>>& result = *new std::vector<std::tuple<Left, Right>>();

		// Scan every element in the other input
		for (unsigned int i = 0; i < otherInput->size(); i++) {
			const Scanned& otherElement = otherInput->at(i);
			Comparable key = otherSelector(otherElement);
			auto& localResults = hashTable.find(key);

			// Join with every hashed row with the same key
			while (localResults != hashTable.end() && localResults->first == key) {
				result.push_back(MakeTuple(localResults->second, otherElement));
				localResults++;
			}
		}
#if DEBUG
		std::cout << "Joining using hash table took " << GetElapsedTime(start) << "ms\n";
#endif
		return result;
	}

public:
	CPUHashJoin(const std::vector<Left>& leftData, const std::vector<Right>& rightData) {
		_leftData = &leftData;
		_rightData = &rightData;
	};

	template <typename Comparable>
	std::vector<std::tuple<Left, Right>>& Join(Comparable(*leftSelector)(const Left), Comparable(*rightSelector)(const Right)) {
		// Always create a hash table for the smaller relation
		if (_leftData->size() <= _rightData->size()) {
			std::unordered_multimap<Comparable, const Left&> hashTable;
			CreateHashTable<Comparable, Left>(_leftData, leftSelector, hashTable);
			return PerformJoin(hashTable, _rightData, rightSelector);
		}
		else {
			std::unordered_multimap<Comparable, const Right&> hashTable;
			CreateHashTable<Comparable, Right>(_rightData, rightSelector, hashTable);
			return PerformJoin(hashTable, _leftData, leftSelector);
		}
	};
};
