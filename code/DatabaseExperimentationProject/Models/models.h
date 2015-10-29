#include <ctime>
#include <iostream>

#ifndef LINEITEM_H
#define LINEITEM_H
struct LineItem {
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

	friend std::ostream &operator<<(std::ostream &output, const LineItem &item) {
		output << "LineItem: {Order key: " << item.order_key << "; Part key: " << item.part_key << "; Supp key: " << item.supp_key << "}";
		return output;
	}
};
#endif

#ifndef ORDERS_H
#define ORDERS_H
struct Order {
	int order_key;
	int customer_key;
	char order_status;
	double total_price;
	tm order_date;
	char order_priority[16];
	char clerk[16];
	int ship_priority;
	char comment[80];

	friend std::ostream &operator<<(std::ostream &output, const Order &order) {
		output << "Order: {Order key: " << order.order_key << "; Customer key: " << order.customer_key << "}";
		return output;
	}
};
#endif
