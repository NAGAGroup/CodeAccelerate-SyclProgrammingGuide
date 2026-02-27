#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

constexpr size_t Group_size = 128;

int main() {
    const std::size_t input_size = 1024;
    std::vector<int> data(input_size);
    for (int i = 0; i < (int)input_size; ++i) data[i] = i;
    
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    
    // Buffer scope: writeback to data[] fires when buff goes out of scope
    {
        auto buff = sycl::make_sync_writeback_view<int>(data.data(), sycl::range<1>{input_size});
        
        q.submit([&](sycl::handler& cgh) {
            auto acc = buff.get_access<sycl::access::mode::read_write>(cgh);
            
            cgh.parallel<class ScopedReductionKernel>(sycl::range<1>{input_size / Group_size}, sycl::range<1>{Group_size}, [=](auto grp) {
                sycl::memory_environment(grp, sycl::require_local_mem<int[Group_size]>(), sycl::require_private_mem<int>(), [&](auto& scratch, auto& private_mem) {
                    sycl::distribute_items(grp, [&](sycl::s_item<1> idx) {
                        scratch[idx.get_local_id(grp, 0)] = acc[idx.get_global_id(0)];
                    });
                    
                    sycl::group_barrier(grp);
                    
                    for (int i = Group_size / 2; i > 0; i /= 2) {
                        sycl::distribute_items_and_wait(grp, [&](sycl::s_item<1> idx) {
                            size_t lid = idx.get_innermost_local_id(0);
                            if (lid < (size_t)i) scratch[lid] += scratch[lid + i];
                        });
                    }
                    
                    sycl::single_item(grp, [&]() {
                        acc[grp.get_group_id(0) * Group_size] = scratch[0];
                    });
                }); // end memory_environment
            }); // end parallel
        }); // end submit
    } // buff goes out of scope here, writeback fires
    
    // Verify results
    bool ok = true;
    for (int grp = 0; grp < (int)(input_size / Group_size); ++grp) {
        int expected = 0;
        for (int k = grp * (int)Group_size; k < (grp + 1) * (int)Group_size; ++k) {
            expected += k;
        }
        
        if (data[grp * Group_size] != expected) {
            std::cout << "Wrong result for group " << grp << ": got " << data[grp * Group_size] << ", expected " << expected << std::endl;
            ok = false;
        }
    }
    
    if (ok) {
        std::cout << "Scoped reduction: OK" << std::endl;
        return 0;
    } else {
        std::cout << "Scoped reduction: FAIL" << std::endl;
        return 1;
    }
}