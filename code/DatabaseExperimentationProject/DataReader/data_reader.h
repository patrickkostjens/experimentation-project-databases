#include "stdafx.h"
#include "vector"
#include "line_item_parser.h"

#ifdef DATAREADER_EXPORTS
#define DATAREADER_API __declspec(dllexport) 
#else
#define DATAREADER_API __declspec(dllimport) 
#endif

DATAREADER_API std::vector<LineItem> ReadAllLineItems(std::string file_path);
