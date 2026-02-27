#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

constexpr int GROUP_SIZE = 256;

int main() {
    const size_t N = 1024 * 1024; // 1M elements
    
    // Create input data: 1.0, 2.0, 3.0, ..., N.0
    std::vector<float> input(N);
    std::iota(input.begin(), input.end(), 1.0f);
    
    // Expected sum: N * (N + 1) / 2
    float expected_sum = static_cast<float>(N) * (static_cast<float>(N) + 1.0f) / 2.0f;
    
    // Create queue with in_order property
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    
    const size_t num_groups = N / GROUP_SIZE;
    
    // Allocate device memory for partial sums
    float* d_output = sycl::malloc_device<float>(num_groups, q);
    float* d_input = sycl::malloc_device<float>(N, q);
    
    // Copy input data to device
    q.memcpy(d_input, input.data(), N * sizeof(float)).wait();
    
    // Submit kernel using nd_range parallel_for
    q.submit([&](sycl::handler& h) {
        // Declare local accessor inside the command group
        sycl::local_accessor<float, 1> scratch(GROUP_SIZE, h);
        
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{GROUP_SIZE}}, 
            [=](sycl::nd_item<1> it) {
                size_t gid = it.get_global_id(0);
                size_t lid = it.get_local_id(0);
                
                // Load data into local memory
                scratch[lid] = d_input[gid];
                
                // Synchronize within work group
                sycl::group_barrier(it.get_group());
                
                // Tree reduction in local memory
                for (int stride = GROUP_SIZE / 2; stride > 0; stride /= 2) {
                    if (lid < static_cast<size_t>(stride)) {
                        scratch[lid] += scratch[lid + stride];
                    }
                    sycl::group_barrier(it.get_group());
                }
                
                // Write partial sum from first work item in group
                if (lid == 0) {
                    d_output[it.get_group_linear_id()] = scratch[0];
                }
            });
    }).wait();
    
    // Copy partial sums back to host
    std::vector<float> partial_sums(num_groups);
    q.memcpy(partial_sums.data(), d_output, num_groups * sizeof(float)).wait();
    
    // Final reduction on host
    float result = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0f);
    
    // Compare with expected (within 1% tolerance)
    float tolerance = expected_sum * 0.01f; // 1%
    bool success = std::abs(result - expected_sum) <= tolerance;
    
    std::cout << "nd_range reduction: " << (success ? "OK" : "FAIL") << "  sum=" << result << std::endl;
    std::cout << "Expected: " << expected_sum << std::endl;
    
    // Free device memory
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    
    return success ? 0 : 1;
}