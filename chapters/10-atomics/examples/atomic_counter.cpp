#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

using namespace sycl;

int main() {
  const size_t N = 1024;
  const size_t num_groups = 16;
  const size_t group_size = 64;
  
  queue q{default_selector_v, property_list{property::queue::in_order{}}};
  
  // Broken version: non-atomic counter with race condition
  {
    int broken_count = 0;
    {
      auto buf = make_sync_writeback_view(&broken_count, range<1>{1});
      q.submit([&](handler& cgh) {
        auto acc = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for(range<1>{N}, [=](id<1> idx) {
          // This is NOT atomic - race condition!
          acc[0]++;  // Each work-item increments without synchronization
        });
      });
      q.wait();
    } // buf destroyed = writeback
    
    std::cout << "Non-atomic count: " << broken_count 
              << " (expected 1024, probably wrong)" << std::endl;
  }
  
  // Correct version: atomic counter
  {
    int correct_count = 0;
    {
      auto buf = make_sync_writeback_view(&correct_count, range<1>{1});
      q.submit([&](handler& cgh) {
        auto acc = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for(range<1>{N}, [=](id<1> idx) {
          // Proper atomic increment
          atomic_ref<int, memory_order::relaxed, memory_scope::device, 
                     access::address_space::global_space> ref{acc[0]};
          ref.fetch_add(1);
        });
      });
      q.wait();
    } // buf destroyed = writeback
    
    std::cout << "Atomic count: " << correct_count 
              << " (expected 1024)" << std::endl;
    
    if (correct_count == 1024) {
      std::cout << "PASS: Atomic counter produced correct result" << std::endl;
    } else {
      std::cout << "FAIL: Atomic counter produced incorrect result" << std::endl;
    }
  }
  
  return 0;
}