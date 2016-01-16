#include <time.h>
#include <string>

const tm ParseDate(const std::string& date_string);

void ReadUntilSeparator(std::ifstream& in_file, char* out, const int len);

const tm ReadDate(std::ifstream& in_file);

int random(int i);
