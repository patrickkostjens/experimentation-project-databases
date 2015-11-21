#ifdef BASICGPUPROCESSOR_EXPORTS
#define BASICGPUPROCESSOR_API __declspec(dllexport) 
#else
#define BASICGPUPROCESSOR_API __declspec(dllimport) 
#endif

template<typename Left, typename Right>
BASICGPUPROCESSOR_API std::vector<std::tuple<Left, Right>>& gpu_hash_join(std::vector<Left>& leftItems, std::vector<Right>& rightItems);

template BASICGPUPROCESSOR_API std::vector<std::tuple<Order, LineItem>>& gpu_hash_join(std::vector<Order>& leftItems, std::vector<LineItem>& rightItems);
