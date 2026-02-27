#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

int main() {
    const size_t N = 16 * 1024 * 1024; // 16M floats
    const int NUM_RUNS = 5;
    
    // Allocate host memory
    std::vector<float> a(N), b(N), c(N);
    std::fill(a.begin(), a.end(), 1.0f);
    std::fill(b.begin(), b.end(), 2.0f);
    
    // Create SYCL queue
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    
    // Print benchmark info
    double data_size_MB = static_cast<double>(N * sizeof(float)) / (1024 * 1024);
    std::cout << "Running bandwidth benchmark: " << N << " floats (" 
              << data_size_MB << " MB per array)" << std::endl;
    
    // Warmup run (not timed)
    {
        auto buf_a = sycl::make_sync_view(a.data(), sycl::range<1>{N});
        auto buf_b = sycl::make_sync_view(b.data(), sycl::range<1>{N});
        auto buf_c = sycl::make_async_writeback_view(c.data(), sycl::range<1>{N}, q);
        
        q.submit([&](sycl::handler& cgh) {
            auto acc_a = buf_a.get_access<sycl::access_mode::read>(cgh);
            auto acc_b = buf_b.get_access<sycl::access_mode::read>(cgh);
            auto acc_c = buf_c.get_access<sycl::access_mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
                acc_c[idx] = acc_a[idx] + acc_b[idx];
            });
        });
        q.wait();
    } // buf_c destroyed here, writeback occurs
    
    // Verify warmup results
    if (c[0] == 3.0f && c[N-1] == 3.0f) {
        std::cout << "Warmup: OK" << std::endl;
    } else {
        std::cout << "Warmup: FAILED" << std::endl;
        return 1;
    }
    
    // Timed runs
    std::vector<double> run_times_ms;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        double elapsed_ms = 0.0;
        {
            auto buf_a = sycl::make_sync_view(a.data(), sycl::range<1>{N});
            auto buf_b = sycl::make_sync_view(b.data(), sycl::range<1>{N});
            auto buf_c = sycl::make_async_writeback_view(c.data(), sycl::range<1>{N}, q);
            
            auto t0 = std::chrono::high_resolution_clock::now();
            
            q.submit([&](sycl::handler& cgh) {
                auto acc_a = buf_a.get_access<sycl::access_mode::read>(cgh);
                auto acc_b = buf_b.get_access<sycl::access_mode::read>(cgh);
                auto acc_c = buf_c.get_access<sycl::access_mode::write>(cgh);
                
                cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
                    acc_c[idx] = acc_a[idx] + acc_b[idx];
                });
            });
            q.wait();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        } // buf_c destroyed here, writeback occurs
        
        run_times_ms.push_back(elapsed_ms);
        std::cout << "Run " << (run + 1) << ": " << elapsed_ms << " ms" << std::endl;
    }
    
    // Calculate average and bandwidth
    double avg_ms = 0.0;
    for (double time : run_times_ms) {
        avg_ms += time;
    }
    avg_ms /= NUM_RUNS;
    
    double bytes = 3.0 * N * sizeof(float); // 3 arrays * N * 4 bytes
    double gb_per_s = bytes / (avg_ms / 1000.0) / 1e9;
    
    std::cout << "Average: " << avg_ms << " ms | Bandwidth: " << gb_per_s << " GB/s" << std::endl;
    
    // [!NOTE]: First run may be slower due to JIT compilation. Run with ACPP_ADAPTIVITY_LEVEL=2 and repeat for best performance.
    
    return 0;
}