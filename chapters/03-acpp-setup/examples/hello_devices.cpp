#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <string>

std::string device_type_to_string(sycl::info::device_type type) {
    switch (type) {
        case sycl::info::device_type::cpu:
            return "CPU";
        case sycl::info::device_type::gpu:
            return "GPU";
        case sycl::info::device_type::accelerator:
            return "Accelerator";
        case sycl::info::device_type::custom:
            return "Custom";
        case sycl::info::device_type::automatic:
            return "Automatic";
        case sycl::info::device_type::all:
            return "All";
        case sycl::info::device_type::host:
            return "Host";
        default:
            return "Unknown";
    }
}

int main() {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    
    std::cout << "AdaptiveCpp Device Enumeration" << std::endl;
    std::cout << "================================" << std::endl;
    
    for (size_t p_idx = 0; p_idx < platforms.size(); ++p_idx) {
        const auto& platform = platforms[p_idx];
        
        std::cout << "Platform " << p_idx << ": " 
                  << platform.get_info<sycl::info::platform::name>() << std::endl;
        std::cout << "  Vendor: " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
        
        std::vector<sycl::device> devices = platform.get_devices();
        
        for (size_t d_idx = 0; d_idx < devices.size(); ++d_idx) {
            const auto& device = devices[d_idx];
            
            std::cout << "  Device " << d_idx << ": " 
                      << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "    Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
            std::cout << "    Type: " << device_type_to_string(device.get_info<sycl::info::device::device_type>()) << std::endl;
            std::cout << "    Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
            std::cout << "    Max Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
            std::cout << "    Global Memory: " << device.get_info<sycl::info::device::global_mem_size>() / 1024 / 1024 << " MiB" << std::endl;
            std::cout << "    Local Memory: " << device.get_info<sycl::info::device::local_mem_size>() / 1024 << " KiB" << std::endl;
        }
        
        if (p_idx < platforms.size() - 1) {
            std::cout << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // Demonstrate queue construction
    sycl::queue q(sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order()});
    std::cout << "Default device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    return 0;
}