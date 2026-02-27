#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

using namespace sycl;

int main() {
  const size_t N = 1024;
  const size_t num_groups = 16;
  const size_t group_size = 64;
  
  queue q{default_selector_v, property_list{property::queue::in_order{}}};
  
  // Initialize data with integer values (1 each) for correct sum = N = 1024
  std::vector<int> data(N, 1);
  
  // Use make_sync_view for input data (read-only)
  auto data_buf = make_sync_view(data.data(), range<1>{N});
  
  // Use make_sync_writeback_view for result (1 element, init 0)
  int result = 0;
  {
    auto result_buf = make_sync_writeback_view(&result, range<1>{1});
    
    q.submit([&](handler& cgh) {
      auto data_acc = data_buf.get_access<access::mode::read>(cgh);
      auto result_acc = result_buf.get_access<access::mode::read_write>(cgh);
      
      cgh.parallel_for(nd_range<1>{range<1>{N}, range<1>{group_size}}, [=](nd_item<1> item) {
        size_t global_id = item.get_global_id(0);
        
        // Each work-item accumulates its assigned element
        // In this simple case, each work-item handles 1 element
        int partial = data_acc[global_id];
        
        // Atomic fetch_add to global result
        atomic_ref<int, memory_order::relaxed, memory_scope::device, 
                   access::address_space::global_space> ref{result_acc[0]};
        ref.fetch_add(partial);
      });
    });
    q.wait();
  } // result_buf destroyed = writeback
  
  std::cout << "Reduction result: " << result << " (expected 1024)" << std::endl;
  
  if (result == 1024) {
    std::cout << "PASS: Integer reduction produced correct result" << std::endl;
  } else {
    std::cout << "FAIL: Integer reduction produced incorrect result" << std::endl;
  }
  
  return 0;
}