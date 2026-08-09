cmake_minimum_required(VERSION 3.22.1)
project(genie_tts_native)

# C++17 for std::vector etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find log library (Android)
find_library(log-lib log)

# The production Android target uses the Java ORT array decoder. The only
# native component kept in the CPU APK is the parity-tested audio resampler.
add_library(native_resampler SHARED
    reference_resampler.cpp
)
target_link_libraries(native_resampler ${log-lib})

include(CTest)

if(BUILD_TESTING AND NOT ANDROID)
    add_executable(reference_resampler_test
        reference_resampler.cpp
        reference_resampler_test.cpp
    )
    target_include_directories(reference_resampler_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME reference_resampler_test COMMAND reference_resampler_test)
endif()
