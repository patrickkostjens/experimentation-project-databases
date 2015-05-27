#include "ctime"

#ifndef LINEITEM_H
#define LINEITEM_H
struct LineItem{
	int order_key;
	int part_key;
	int supp_key;
	int line_number;
	double quantity;
	double extended_price;
	double discount;
	double tax;
	char return_flag;
	char line_status;
	tm ship_date;
	tm commit_date;
	tm receipt_date;
	char ship_instruct[25];
	char ship_mode[10];
	char comment[44];
};
#endif