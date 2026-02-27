#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>

int main() {
    const size_t N = 1024;
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    
    // Pattern A - make_sync_buffer (internal storage, no host pointer needed)
    {
        auto internal_buf = sycl::make_sync_buffer<float>(sycl::range<1>{N});
        
        q.submit([&](sycl::handler& cgh) {
            auto acc = internal_buf.get_access<sycl::access_mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
                acc[i] = static_cast<float>(i[0]);
            });
        });
        
        // Verify with host accessor
        auto host_acc = internal_buf.get_access<sycl::access_mode::read>();
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (host_acc[i] != static_cast<float>(i)) {
                ok = false;
                break;
            }
        }
        std::cout << "Pattern A: " << (ok ? "OK" : "FAILED") << std::endl;
    }
    
    // Pattern B - make_sync_writeback_view (external storage, writeback on destroy)
    {
        std::vector<float> host_data(N, 0.0f);
        
        {
            auto writeback_buf = sycl::make_sync_writeback_view<float>(host_data.data(), sycl::range<1>{N});
            
            q.submit([&](sycl::handler& cgh) {
                auto acc = writeback_buf.get_access<sycl::access_mode::write>(cgh);
                
                cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
                    acc[i] = 2.0f;
                });
            });
        } // Buffer goes out of scope, writes back to host_data
        
        // Verify host_data was written back
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (host_data[i] != 2.0f) {
                ok = false;
                break;
            }
        }
        std::cout << "Pattern B: " << (ok ? "OK" : "FAILED") << std::endl;
    }
    
    // Pattern C - make_sync_view (external, no writeback, read-only use case)
    {
        std::vector<float> input(N, 3.0f);
        auto input_buf = sycl::make_sync_view<const float>(input.data(), sycl::range<1>{N});
        auto output_buf = sycl::make_sync_buffer<float>(sycl::range<1>{N});
        
        q.submit([&](sycl::handler& cgh) {
            auto in_acc = input_buf.get_access<sycl::access_mode::read>(cgh);
            auto out_acc = output_buf.get_access<sycl::access_mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
                out_acc[i] = in_acc[i] * 2.0f;
            });
        });
        
        // Verify the result
        auto host_acc = output_buf.get_access<sycl::access_mode::read>();
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (host_acc[i] != 6.0f) { // 3.0f * 2.0f
                ok = false;
                break;
            }
        }
        std::cout << "Pattern C: " << (ok ? "OK" : "FAILED") << std::endl;
    }
    
    return 0;
}