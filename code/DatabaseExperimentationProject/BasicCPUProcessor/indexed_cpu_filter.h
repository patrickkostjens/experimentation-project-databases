#include <iostream>
#include <ctime>
#include "b_tree.h"

template <class ObjectType, typename FilterColumnType>
class IndexedCPUFilter {
private:
	const std::vector<ObjectType> *_data;

	inline double GetElapsedTime(clock_t& since) {
		return (std::clock() - since) / (double)CLOCKS_PER_SEC * 1000;
	}
public:
	IndexedCPUFilter(const std::vector<ObjectType>& data) {
		_data = &data;
		return;
	};

	std::vector<ObjectType>& Filter(FilterColumnType(*selector)(const ObjectType), FilterColumnType searchValue) {
#if DEBUG
		std::cout << "Starting building index\n";
#endif
		std::clock_t start = std::clock();
		BTree<FilterColumnType, ObjectType> index(3);

		for (unsigned int i = 0; i < _data->size(); i++) {
			index.insert(selector(_data->at(i)), _data->at(i));
		}
		double buildTime = GetElapsedTime(start);
#if DEBUG
		std::cout << "Building index took " << buildTime << "ms\n";
#endif
		start = std::clock();

		std::vector<ObjectType> *result = new std::vector<ObjectType>(index.find(searchValue));
		double searchTime = GetElapsedTime(start);
#if DEBUG
		std::cout << "Indexed search took " << searchTime << "ms\n";
#endif

		std::cout << buildTime << " " << searchTime << "\n";
		return *result;
	};
};
