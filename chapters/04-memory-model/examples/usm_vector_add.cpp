#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 1024 * 1024; // 1M elements
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    
    // Step 1: Allocate three device arrays
    float* a = sycl::malloc_device<float>(N, q);
    float* b = sycl::malloc_device<float>(N, q);
    float* c = sycl::malloc_device<float>(N, q);
    
    // Step 2: Allocate host staging buffers
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);
    
    // Step 3: Copy h_a -> a and h_b -> b
    q.memcpy(a, h_a.data(), N * sizeof(float));
    q.memcpy(b, h_b.data(), N * sizeof(float));
    q.wait(); // Ensure copies complete
    
    // Step 4: Submit parallel_for kernel
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
            c[i] = a[i] + b[i];
        });
    });
    
    // Step 5: Copy result back
    q.memcpy(h_c.data(), c, N * sizeof(float));
    q.wait();
    
    // Step 6: Verify h_c[i] == 3.0f for all i
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) {
            ok = false;
            break;
        }
    }
    std::cout << "USM vector add: " << (ok ? "OK" : "FAILED") << std::endl;
    
    // Step 7: Free device memory
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    
    return 0;
}