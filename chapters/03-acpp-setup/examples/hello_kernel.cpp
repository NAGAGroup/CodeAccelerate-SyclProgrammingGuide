#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <chrono>

int main(int argc, char* argv[]) {
    // Parse optional command-line argument for N
    size_t N = 1024 * 1024;  // Default value
    if (argc > 1) {
        try {
            N = std::stoull(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing N: " << e.what() << std::endl;
            return 1;
        }
    }
    
    // Create input vectors
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N, 0.0f);  // Results
    
    // Fill vectors with test data
    std::iota(a.begin(), a.end(), 0.0f);
    std::fill(b.begin(), b.end(), 1.0f);
    
    // Create queue
    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    // Time the kernel execution
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use scoped block to control buffer lifetimes
    {
        // Create buffers using AdaptiveCpp buffer factories
        auto buf_a = sycl::make_async_writeback_view(a.data(), sycl::range<1>{N}, q);
        auto buf_b = sycl::make_sync_view(b.data(), sycl::range<1>{N});
        auto buf_c = sycl::make_async_writeback_view(c.data(), sycl::range<1>{N}, q);
        
        // Submit kernel
        q.submit([&](sycl::handler& cgh) {
            auto acc_a = buf_a.get_access<sycl::access_mode::read>(cgh);
            auto acc_b = buf_b.get_access<sycl::access_mode::read>(cgh);
            auto acc_c = buf_c.get_access<sycl::access_mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
                acc_c[idx] = acc_a[idx] + acc_b[idx];
            });
        });
    }
    
    // Wait for async writebacks to complete
    q.wait();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float time_ms = duration.count() / 1000.0f;
    
    // Verify results
    const float epsilon = 1e-5f;
    bool passed = true;
    
    // Check first and last 3 elements
    std::cout << "First 3 elements: ";
    for (size_t i = 0; i < 3; ++i) {
        float expected = static_cast<float>(i) + 1.0f;
        std::cout << c[i] << " (expected " << expected << ") ";
        if (std::abs(c[i] - expected) > epsilon) {
            passed = false;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Last 3 elements: ";
    for (size_t i = N - 3; i < N; ++i) {
        float expected = static_cast<float>(i) + 1.0f;
        std::cout << c[i] << " (expected " << expected << ") ";
        if (std::abs(c[i] - expected) > epsilon) {
            passed = false;
        }
    }
    std::cout << std::endl;
    
    // Check all elements
    for (size_t i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) + 1.0f;
        if (std::abs(c[i] - expected) > epsilon) {
            passed = false;
            break;
        }
    }
    
    std::cout << "Verification: " << (passed ? "PASS" : "FAIL") << std::endl;
    
    // Print throughput
    float bytes_processed = 3 * N * sizeof(float);  // 2 reads + 1 write
    float gb_per_s = (bytes_processed / 1024.0f / 1024.0f / 1024.0f) / (time_ms / 1000.0f);
    
    std::cout << "Processed " << N << " elements in " << time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gb_per_s << " GB/s" << std::endl;
    
    return passed ? 0 : 1;
}