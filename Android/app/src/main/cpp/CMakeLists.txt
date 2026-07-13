cmake_minimum_required(VERSION 3.22.1)
project(native_decoder_loop)

# C++17 for std::vector etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the native decoder loop library
add_library(native_decoder_loop SHARED
    native_decoder_loop.cpp
    native_semantic_slicing.cpp
)

# Find log library (Android)
find_library(log-lib log)

# Link against the prebuilt onnxruntime shared library from the AAR.
# The AAR ships libonnxruntime.so under jni/{ABI}/ during the build.
# We find it via the ORT_ANDROID_QNN_AAR environment or by searching.
# Since the AAR is a dependency, AGP extracts .so files to nativeLibs.
# We just need to declare it as imported.
set(ORT_LIB_DIR "${CMAKE_SOURCE_DIR}/../../../build/intermediates/merged_native_libs/debug/obj/${ANDROID_ABI}")

# Fallback: search common paths
if(NOT EXISTS "${ORT_LIB_DIR}/libonnxruntime.so")
    file(GLOB_RECURSE ORT_SO_FILES
        "${CMAKE_SOURCE_DIR}/../../../**/libonnxruntime.so"
    )
    if(ORT_SO_FILES)
        list(GET ORT_SO_FILES 0 ORT_LIB_DIR)
        get_filename_component(ORT_LIB_DIR "${ORT_LIB_DIR}" DIRECTORY)
    endif()
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB_DIR}/libonnxruntime.so"
)

target_link_libraries(native_decoder_loop
    ${log-lib}
    onnxruntime
)

target_include_directories(native_decoder_loop PRIVATE
    ${CMAKE_SOURCE_DIR}
)

include(CTest)

if(BUILD_TESTING AND NOT ANDROID)
    add_executable(native_semantic_slicing_test
        native_semantic_slicing.cpp
        native_semantic_slicing_test.cpp
    )
    target_include_directories(native_semantic_slicing_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME native_semantic_slicing_test COMMAND native_semantic_slicing_test)
endif()
