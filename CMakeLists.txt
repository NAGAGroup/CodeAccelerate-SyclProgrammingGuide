cmake_minimum_required(VERSION 3.20)

project(acpp-tutorial CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)

# Helper macro to add AdaptiveCpp examples
macro(add_acpp_example name source)
    add_executable(${name} ${source})
    add_sycl_to_target(TARGET ${name})
    install(TARGETS ${name} DESTINATION bin)
endmacro()

# Add subdirectories for chapters with examples
add_subdirectory(chapters/03-acpp-setup/examples)
add_subdirectory(chapters/04-memory-model/examples)
add_subdirectory(chapters/05-kernels-and-parallelism/examples)
add_subdirectory(chapters/06-acpp-extensions/examples)
add_subdirectory(chapters/07-performance/examples)