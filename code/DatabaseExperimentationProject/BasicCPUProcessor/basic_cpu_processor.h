#include "vector"

template <class T>
class BasicCPUProcessor {
private:
	std::vector<T> *data_;
public:
	BasicCPUProcessor(std::vector<T> data){
		data_ = new std::vector<T>(data);
		return;
	};

	std::vector<T> Filter(bool (*filter)(T)){
		std::vector<T> *result = new std::vector<T>();
		for (unsigned int i = 0; i < data_->size(); i++) {
			if (filter((*data_)[i])) {
				result->push_back((*data_)[i]);
			}
		}
		return *result;
	};
};