#include "stdafx.h"

#ifdef DATAREADER_EXPORTS
#define DATAREADER_API __declspec(dllexport) 
#else
#define DATAREADER_API __declspec(dllimport) 
#endif

DATAREADER_API std::vector<LineItem>& ReadAllLineItems(const std::string file_path);
DATAREADER_API std::vector<Order>& ReadAllOrders(const std::string file_path);
