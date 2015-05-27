// SimpleCPUProcessor.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "basic_cpu_processor.h"

template <class T>
BasicCPUProcessor<T>::BasicCPUProcessor(std::vector<T> data)
{
	data_ = new std::vector<T>(data);
	return;
}

template <class T>
T BasicCPUProcessor<T>::GetFirst() {
	return (*data_)[0];
}
