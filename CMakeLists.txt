cmake_minimum_required(VERSION 3.20)

project(voxel VERSION 1.0.0 LANGUAGES CXX)

option(VOXEL_BUILD_SHARED "Build voxel as a shared library" OFF)
option(VOXEL_BUILD_EXAMPLES "Build example programs" ON)
option(VOXEL_BUILD_PYTHON "Build Python bindings" ON)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Header-only library — use an INTERFACE target (no source compilation needed)
add_library(voxel INTERFACE)

target_include_directories(voxel INTERFACE include/)

target_compile_features(voxel INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(voxel INTERFACE /W4 /Oi)
else()
    target_compile_options(voxel INTERFACE -Wall -Wextra -O3 -march=native)
endif()

if(VOXEL_BUILD_EXAMPLES)
    add_executable(voxel-demo examples/demo.cpp)
    target_link_libraries(voxel-demo PRIVATE voxel)
    add_executable(voxel-test examples/test_suite.cpp)
    target_link_libraries(voxel-test PRIVATE voxel)
    add_executable(voxel-jit-test examples/jit_test.cpp)
    target_link_libraries(voxel-jit-test PRIVATE voxel)

    if(MSVC)
        target_compile_options(voxel-demo PRIVATE /W4 /Oi)
        target_compile_options(voxel-test PRIVATE /W4 /Oi)
        target_compile_options(voxel-jit-test PRIVATE /W4 /Oi)
    else()
        target_compile_options(voxel-demo PRIVATE -Wall -Wextra -O3 -march=native)
        target_compile_options(voxel-test PRIVATE -Wall -Wextra -O3 -march=native)
        target_compile_options(voxel-jit-test PRIVATE -Wall -Wextra -Wno-free-nonheap-object -O3 -march=native)
    endif()
endif()

enable_testing()
add_test(NAME voxel-test  COMMAND voxel-test)
add_test(NAME voxel-jit   COMMAND voxel-jit-test)

install(DIRECTORY include/voxel DESTINATION include)

if(VOXEL_BUILD_PYTHON)
    pybind11_add_module(voxel_py bindings/voxel_bindings.cpp)
    target_link_libraries(voxel_py PRIVATE voxel)
    target_compile_features(voxel_py PRIVATE cxx_std_20)
    if(NOT MSVC)
        target_compile_options(voxel_py PRIVATE -Wall -Wextra -O3 -march=native)
    endif()
endif()

# C ABI shared library (for Go cgo, Java FFM, and any C-compatible FFI)
option(VOXEL_BUILD_C_ABI "Build C ABI shared library" ON)
if(VOXEL_BUILD_C_ABI)
    add_library(voxel_c SHARED bindings/voxel_c.cpp)
    target_link_libraries(voxel_c PRIVATE voxel)
    target_compile_features(voxel_c PRIVATE cxx_std_20)
    target_include_directories(voxel_c PRIVATE ${CMAKE_SOURCE_DIR})
    if(NOT MSVC)
        target_compile_options(voxel_c PRIVATE -Wall -Wextra -O3 -march=native)
    endif()
endif()
