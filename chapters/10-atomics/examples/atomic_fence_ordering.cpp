#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

using namespace sycl;

int main() {
  const size_t N = 64;
  const size_t num_producers = 32;
  
  queue q{default_selector_v, property_list{property::queue::in_order{}}};
  
  auto sd_buf = make_async_buffer<int>(range<1>{N});
  auto flags_buf = make_async_buffer<int>(range<1>{N/2});
  
  // Initialize both to zero
  q.submit([&](handler& cgh){
    auto acc = sd_buf.get_access<access::mode::write>(cgh);
    cgh.fill(acc, 0);
  });
  
  q.submit([&](handler& cgh){
    auto acc = flags_buf.get_access<access::mode::write>(cgh);
    cgh.fill(acc, 0);
  });
  q.wait();
  
  int output_val = 0;
  {
    auto out_buf = make_sync_writeback_view(&output_val, range<1>{1});
    
    // Producer kernel submit
    q.submit([&](handler& cgh) {
      auto sd_acc = sd_buf.get_access<access::mode::read_write>(cgh);
      auto flags_acc = flags_buf.get_access<access::mode::read_write>(cgh);
      
      cgh.parallel_for(nd_range<1>{num_producers, num_producers}, [=](nd_item<1> idx) {
        size_t id = idx.get_global_id();
        
        // Write data
        sd_acc[id] = static_cast<int>(id + 1);
        
        // Release fence ensures data write completes before flag store
        atomic_fence(memory_order::release, memory_scope::device);
        
        // Set flag to signal data is ready
        atomic_ref<int, memory_order::acq_rel, memory_scope::device, 
                   access::address_space::global_space> flag_ref{flags_acc[id]};
        flag_ref.store(1);
      });
    });
    
    // Consumer kernel submit
    q.submit([&](handler& cgh) {
      auto sd_acc = sd_buf.get_access<access::mode::read>(cgh);
      auto flags_acc = flags_buf.get_access<access::mode::read_write>(cgh);
      auto out_acc = out_buf.get_access<access::mode::read_write>(cgh);
      
      cgh.parallel_for(nd_range<1>{num_producers, num_producers}, [=](nd_item<1> idx) {
        size_t id = idx.get_global_id();
        
        // Wait for producer flag
        atomic_ref<int, memory_order::acq_rel, memory_scope::device, 
                   access::address_space::global_space> flag_ref{flags_acc[id]};
        int f = flag_ref.load();
        
        // Acquire fence ensures flag load completes before data read
        atomic_fence(memory_order::acquire, memory_scope::device);
        
        // Read data
        int val = sd_acc[id];
        
        // Add to output sum
        atomic_ref<int, memory_order::relaxed, memory_scope::device, 
                   access::address_space::global_space> out_ref{out_acc[0]};
        out_ref.fetch_add(val);
      });
    });
    q.wait();
  } // end nested scope
  
  // Expected output sum = 1+2+...+32 = 528
  std::cout << "Fence ordering result: " << output_val << " (expected 528)" << std::endl;
  
  if (output_val == 528) {
    std::cout << "PASS: Producer/consumer with fences produced correct result" << std::endl;
  } else {
    std::cout << "FAIL: Producer/consumer with fences produced incorrect result" << std::endl;
  }
  
  return 0;
}