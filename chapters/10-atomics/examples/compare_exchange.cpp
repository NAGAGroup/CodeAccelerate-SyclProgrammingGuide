#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

using namespace sycl;

int main() {
  const size_t N = 256;
  
  queue q{default_selector_v, property_list{property::queue::in_order{}}};
  
  std::vector<float> host_data(N);
  for (size_t i = 0; i < N; ++i) {
    host_data[i] = static_cast<float>(i % 100);
  }
  
  auto data_buf = make_sync_view(host_data.data(), range<1>{N});
  
  float result = 0.0f;
  {
    auto result_buf = make_sync_writeback_view(&result, range<1>{1});
    
    q.submit([&](handler& cgh) {
      auto data_acc = data_buf.get_access<access::mode::read>(cgh);
      auto result_acc = result_buf.get_access<access::mode::read_write>(cgh);
      
      cgh.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx[0];
        float local_val = data_acc[i];
        
        // CAS loop for atomic max
        float expected = result_acc[0];
        while (local_val > expected) {
          atomic_ref<float, memory_order::acq_rel, memory_scope::device, 
                     access::address_space::global_space> ref{result_acc[0]};
          if (ref.compare_exchange_strong(expected, local_val)) {
            break;  // Successfully updated
          }
          // expected was updated by compare_exchange_strong on failure - retry
        }
      });
    });
    q.wait();
  } // end nested scope - writeback triggers
  
  std::cout << "Atomic max: " << result << " (expected 99)" << std::endl;
  
  if (std::abs(result - 99.0f) < 0.001f) {
    std::cout << "PASS: Atomic max produced correct result" << std::endl;
  } else {
    std::cout << "FAIL: Atomic max produced incorrect result" << std::endl;
  }
  
  return 0;
}