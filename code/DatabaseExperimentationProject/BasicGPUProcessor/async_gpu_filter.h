#ifdef BASICGPUPROCESSOR_EXPORTS
#define BASICGPUPROCESSOR_API __declspec(dllexport) 
#else
#define BASICGPUPROCESSOR_API __declspec(dllimport) 
#endif

template<typename TItem>
BASICGPUPROCESSOR_API std::vector<TItem>& gpu_filter_async(std::vector<TItem>& items);

template BASICGPUPROCESSOR_API std::vector<LineItem>& gpu_filter_async<LineItem>(std::vector<LineItem>& items);
template BASICGPUPROCESSOR_API std::vector<Order>& gpu_filter_async<Order>(std::vector<Order>& items);
