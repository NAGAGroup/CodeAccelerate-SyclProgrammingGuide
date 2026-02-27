#define ACPP_EXT_FP_ATOMICS
#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

using namespace sycl;

int main() {
  const size_t N = 512;
  
  queue q{default_selector_v, property_list{property::queue::in_order{}}};
  
  std::vector<float> data_host(N, 1.0f);
  auto data_buf = make_sync_view(data_host.data(), range<1>{N});
  
  float result = 0.0f;
  {
    auto result_buf = make_sync_writeback_view(&result, range<1>{1});
    
    q.submit([&](handler& cgh) {
      auto data_acc = data_buf.get_access<access::mode::read>(cgh);
      auto result_acc = result_buf.get_access<access::mode::read_write>(cgh);
      
      cgh.parallel_for(range<1>{N}, [=](id<1> idx) {
        size_t i = idx[0];
        
#ifdef ACPP_EXT_FP_ATOMICS
        // Direct fetch_add with FP atomics extension
        atomic_ref<float, memory_order::relaxed, memory_scope::device, 
                   access::address_space::global_space> ref{result_acc[0]};
        ref.fetch_add(data_acc[i]);
#else
        // CAS fallback for portable float reduction
        atomic_ref<float, memory_order::acq_rel, memory_scope::device, 
                   access::address_space::global_space> ref{result_acc[0]};
        float expected = ref.load(memory_order::relaxed);
        float desired;
        do {
          desired = expected + data_acc[i];
        } while (!ref.compare_exchange_weak(expected, desired));
#endif
      });
    });
    q.wait();
  } // end nested scope
  
  std::cout << "FP atomic sum: " << result << " (expected 512.0)" << std::endl;
  
  if (std::abs(result - 512.0f) < 0.01f) {
    std::cout << "PASS: FP atomic sum produced correct result" << std::endl;
  } else {
    std::cout << "FAIL: FP atomic sum produced incorrect result" << std::endl;
  }
  
  return 0;
}