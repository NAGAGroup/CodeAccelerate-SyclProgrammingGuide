#include <sycl/sycl.hpp>
#define ACPP_EXT_EXPLICIT_BUFFER_POLICIES
#include <hipSYCL/sycl/buffer_explicit_behavior.hpp>
#include <iostream>
#include <vector>

int main() {
    const size_t N = 1024 * 1024;
    std::vector<float> data(N, 1.0f);
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    {
        auto data_view = sycl::make_async_writeback_view(data.data(), sycl::range<1>{N}, q);
        q.submit([&](sycl::handler& cgh) {
            auto acc = data_view.get_access<sycl::access_mode::read_write>(cgh);
            sycl::specialized<float> scale{2.0f};
            cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
                acc[idx] = acc[idx] * scale;
            });
        });
        q.wait();
    } // data_view destroyed here - triggers async writeback to data.data()
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (data[i] != 2.0f) { success = false; break; }
    }
    if (success) {
        std::cout << "JIT specialized: OK (scale=2.0 applied to 1M elements)" << std::endl;
        return 0;
    } else {
        std::cout << "JIT specialized: FAILED" << std::endl;
        return 1;
    }
}