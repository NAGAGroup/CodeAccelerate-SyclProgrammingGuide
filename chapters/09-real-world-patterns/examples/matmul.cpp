#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <iostream>
#include <vector>

constexpr int N = 256;
constexpr int K = 256;
constexpr int TILE_SIZE = 16;

int main() {
    try {
        // Initialize host matrices
        std::vector<std::vector<float>> a_host(N, std::vector<float>(K));
        std::vector<std::vector<float>> b_host(K, std::vector<float>(N));
        std::vector<std::vector<float>> c_host(N, std::vector<float>(N));
        
        // Initialize A with a[i][j] = (float)(i+1)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < K; ++j) {
                a_host[i][j] = static_cast<float>(i + 1);
            }
        }
        
        // Initialize B with b[i][j] = (float)(j+1)
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                b_host[i][j] = static_cast<float>(j + 1);
            }
        }
        
        // Create SYCL queue
        sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
        
        // Flatten matrices for buffer access
        std::vector<float> a_flat(N * K);
        std::vector<float> b_flat(K * N);
        std::vector<float> c_flat(N * N);
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < K; ++j) {
                a_flat[i * K + j] = a_host[i][j];
            }
        }
        
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                b_flat[i * N + j] = b_host[i][j];
            }
        }
        
        {
            // Create buffers using factory methods
            auto a_buf = make_sync_view<float>(a_flat.data(), sycl::range<2>{N, K});
            auto b_buf = make_sync_view<float>(b_flat.data(), sycl::range<2>{K, N});
            auto c_buf = make_sync_writeback_view<float>(c_flat.data(), sycl::range<2>{N, N});
            
            // Submit kernel
            q.submit([&](sycl::handler& cgh) {
                // Create accessors inside the command group
                auto a_acc = sycl::accessor{a_buf, cgh, sycl::read_only};
                auto b_acc = sycl::accessor{b_buf, cgh, sycl::read_only};
                auto c_acc = sycl::accessor{c_buf, cgh, sycl::read_write};
                
                // Create local memory tiles
                sycl::local_accessor<float, 2> a_tile{sycl::range<2>{TILE_SIZE, TILE_SIZE}, cgh};
                sycl::local_accessor<float, 2> b_tile{sycl::range<2>{TILE_SIZE, TILE_SIZE}, cgh};
                
                // Define nd_range
                sycl::nd_range<2> range{sycl::range<2>{N, N}, sycl::range<2>{TILE_SIZE, TILE_SIZE}};
                
                // Kernel
                cgh.parallel_for<class MatMulKernel>(range, [=](sycl::nd_item<2> item) {
                    int row = item.get_global_id(0);
                    int col = item.get_global_id(1);
                    int local_row = item.get_local_id(0);
                    int local_col = item.get_local_id(1);
                    
                    float sum = 0.0f;
                    
                    // Loop over tiles
                    for (int t = 0; t < N / TILE_SIZE; ++t) {
                        // Load tiles into local memory
                        a_tile[local_row][local_col] = a_acc[row][t * TILE_SIZE + local_col];
                        b_tile[local_row][local_col] = b_acc[t * TILE_SIZE + local_row][col];
                        
                        // Barrier: ensure tiles are loaded before computation
                        sycl::group_barrier(item.get_group());
                        
                        // Compute partial dot product
                        for (int k = 0; k < TILE_SIZE; ++k) {
                            sum += a_tile[local_row][k] * b_tile[k][local_col];
                        }
                        
                        // Barrier: ensure computation completes before next tile load
                        sycl::group_barrier(item.get_group());
                    }
                    
                    c_acc[row][col] = sum;
                });
            });
            
            // Wait for kernel completion
            q.wait();
            
            // Buffer destruction triggers writeback for make_sync_writeback_view
        }
        
        // Copy result back to 2D host matrix
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                c_host[i][j] = c_flat[i * N + j];
            }
        }
        
        // Verification
        // Expected: c[i][j] = (i+1)*(j+1)*N
        bool passed = true;
        
        // Check c[0][N-1] = (0+1)*(N-1+1)*N = 1*N*N = N*N = 65536
        if (std::abs(c_host[0][N-1] - static_cast<float>(N * N)) > 1e-4f) {
            std::cout << "FAIL: c[0][N-1] = " << c_host[0][N-1] << ", expected " << N * N << std::endl;
            passed = false;
        }
        
        // Check c[N-1][0] = N*(0+1)*N = N*N = 65536
        if (std::abs(c_host[N-1][0] - static_cast<float>(N * N)) > 1e-4f) {
            std::cout << "FAIL: c[N-1][0] = " << c_host[N-1][0] << ", expected " << N * N << std::endl;
            passed = false;
        }
        
        // Check c[1][1] = (1+1)*(1+1)*N = 4*256 = 1024
        if (std::abs(c_host[1][1] - 1024.0f) > 1e-4f) {
            std::cout << "FAIL: c[1][1] = " << c_host[1][1] << ", expected 1024" << std::endl;
            passed = false;
        }
        
        if (passed) {
            std::cout << "Matrix multiply: OK" << std::endl;
            return 0;
        } else {
            return 1;
        }
        
    } catch (const sycl::exception& e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }
}