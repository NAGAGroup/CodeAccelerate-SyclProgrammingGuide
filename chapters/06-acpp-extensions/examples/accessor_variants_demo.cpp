#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 1024;
    std::vector<int> src(N);
    std::vector<int> dst(N, 0);
    for (size_t i = 0; i < N; ++i) { src[i] = static_cast<int>(i); }
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    {
        auto src_buf = sycl::make_sync_view(src.data(), sycl::range<1>{N});
        auto dst_buf = sycl::make_async_writeback_view(dst.data(), sycl::range<1>{N}, q);
        q.submit([&](sycl::handler& cgh) {
            auto src_acc = src_buf.get_access<sycl::access_mode::read>(cgh);
            auto dst_acc = dst_buf.get_access<sycl::access_mode::write>(cgh);
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
                dst_acc[idx] = src_acc[idx] * 2;
            });
        });
        q.wait();
    } // dst_buf destroyed here - triggers async writeback to dst.data()
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (dst[i] != static_cast<int>(i * 2)) { success = false; break; }
    }
    if (success) {
        std::cout << "Accessor variants demo: OK" << std::endl;
        return 0;
    } else {
        std::cout << "Accessor variants demo: FAILED" << std::endl;
        return 1;
    }
}