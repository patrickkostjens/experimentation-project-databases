template <class T>
class BasicCPUFilter {
private:
	const std::vector<T> *data_;
public:
	BasicCPUFilter(const std::vector<T>& data){
		data_ = &data;
		return;
	};

	std::vector<T>& Filter(bool (*filter)(const T)){
		std::vector<T> *result = new std::vector<T>();
		for (unsigned int i = 0; i < data_->size(); i++) {
			if (filter((*data_)[i])) {
				result->push_back((*data_)[i]);
			}
		}
		return *result;
	};
};
