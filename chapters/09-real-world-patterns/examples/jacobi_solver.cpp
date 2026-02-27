#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <iostream>
#include <vector>
#include <cmath>

constexpr int N = 512;
constexpr int MAX_ITER = 200;

int main() {
    try {
        // Create SYCL queue
        sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
        
        // Create buffers for internal work (no writeback, non-blocking destructor)
        auto x_cur_buf = make_async_buffer<float>(sycl::range<1>{N});
        auto x_new_buf = make_async_buffer<float>(sycl::range<1>{N});
        
        // Initialize x_cur to 0.0f on device
        q.submit([&](sycl::handler& cgh) {
            auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::write_only};
            cgh.parallel_for<class InitKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
                x_cur_acc[i] = 0.0f;
            });
        });
        
        // Allocate USM scalar for norm computation
        float* norm_ptr = sycl::malloc_device<float>(1, q);
        
        std::cout << "Starting Jacobi solver..." << std::endl;
        
        // Main iteration loop
        for (int iter = 0; iter < MAX_ITER; ++iter) {
            // Kernel 1: Update solution
            q.submit([&](sycl::handler& cgh) {
                auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::read_only};
                auto x_new_acc = sycl::accessor{x_new_buf, cgh, sycl::write_only};
                
                cgh.parallel_for<class UpdateKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
                    float new_val = 1.0f;  // RHS b[i] = 1.0f
                    
                    // Subtract left neighbor contribution
                    if (i > 0) {
                        new_val += x_cur_acc[i - 1];  // -(-1) = +1
                    }
                    
                    // Subtract right neighbor contribution  
                    if (i < N - 1) {
                        new_val += x_cur_acc[i + 1];  // -(-1) = +1
                    }
                    
                    // Divide by diagonal element (4.0f)
                    x_new_acc[i] = new_val / 4.0f;
                });
            });
            
            // Kernel 2: Copy new solution back
            // No explicit event dependency needed - AdaptiveCpp DAG tracks
            // accessor conflicts automatically between kernels
            q.submit([&](sycl::handler& cgh) {
                auto x_new_acc = sycl::accessor{x_new_buf, cgh, sycl::read_only};
                auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::write_only};
                
                cgh.parallel_for<class CopyKernel>(sycl::range<1>{N}, [=](sycl::id<1> i) {
                    x_cur_acc[i] = x_new_acc[i];
                });
            });
            
            // Compute norm every 50 iterations
            if (iter % 50 == 0) {
                // Reset norm to 0
                q.memcpy(norm_ptr, &std::vector<float>{0.0f}[0], sizeof(float)).wait();
                
                // Compute norm using reduction
                q.submit([&](sycl::handler& cgh) {
                    auto x_cur_acc = sycl::accessor{x_cur_buf, cgh, sycl::read_only};
                    
                    cgh.parallel_for<class NormKernel>(
                        sycl::range<1>{N}, 
                        sycl::reduction(norm_ptr, sycl::plus<float>()),
                        [=](sycl::id<1> i, auto& sum) {
                            sum += std::abs(x_cur_acc[i]);
                        }
                    );
                });
                
                // Wait and copy norm to host
                q.wait();
                float norm;
                q.memcpy(&norm, norm_ptr, sizeof(float)).wait();
                
                std::cout << "Iteration " << iter << ": norm = " << norm << std::endl;
            }
        }
        
        // Final wait to ensure all kernels complete
        q.wait();
        
        // Free USM memory
        sycl::free(norm_ptr, q);
        
        std::cout << "Jacobi solver: completed " << MAX_ITER << " iterations" << std::endl;
        return 0;
        
    } catch (const sycl::exception& e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }
}