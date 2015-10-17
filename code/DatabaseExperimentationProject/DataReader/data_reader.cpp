// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "data_reader.h"
#include "line_item_parser.h"

std::vector<LineItem>& ReadAllLineItems(const std::string file_path) {
	return ReadAndParseLineItems(file_path);
}
