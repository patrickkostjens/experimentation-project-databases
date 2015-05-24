// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "data_reader.h"

std::vector<LineItem> ReadAllLineItems(std::string file_path) {
	return ReadAndParseLineItems(file_path);
}