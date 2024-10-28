cmake_minimum_required(VERSION 3.12)
project(ann LANGUAGES C CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for architecture without Unix commands
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARCHITECTURE "aarch64")
else()
    set(ARCHITECTURE "x86_64")
endif()
message(STATUS "Architecture: ${ARCHITECTURE}")

# Choose the compiler based on architecture and system
if(ARCHITECTURE STREQUAL "aarch64")
    set(CMAKE_C_COMPILER "/home/dawid/gcc-13.2.0/bin/gcc")
    set(CMAKE_CXX_COMPILER "/home/dawid/gcc-13.2.0/bin/g++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,/home/dawid/gcc-13.2.0/lib64/")
    link_directories("/home/dawid/gcc-13.2.0/lib64/")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -lz -gdwarf-2")
else()
    if(WIN32)
        # Adjust GCC path if using MinGW on Windows
        set(CMAKE_C_COMPILER "D:/Programs/x86_64-14.2.0-release-posix-seh-ucrt-rt_v12-rev0/mingw64/bin/gcc.exe")
        set(CMAKE_CXX_COMPILER "D:/Programs/x86_64-14.2.0-release-posix-seh-ucrt-rt_v12-rev0/mingw64/bin/g++.exe")
    else()
        set(CMAKE_C_COMPILER gcc)
        set(CMAKE_CXX_COMPILER g++)
    endif()
endif()

# Compiler and linker flags
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -falign-jumps=64 -fopenmp -fPIC -Ofast -march=native")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/..)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../third_party/helpa)

# Add the library and executable
add_library(ann STATIC ann.cpp ${PROJECT_SOURCE_DIR}/ann.cpp)
add_executable(run main.cpp)

# Link libraries
if(ARCHITECTURE STREQUAL "aarch64")
    target_link_libraries(run ann ${HDF5_LIB_PATH})
else()
    if(WIN32)
        # Add specific Windows libraries here if needed
        target_link_libraries(run ann)
    else()
        target_link_libraries(run ann ${HDF5_LIB_PATH} dl z)
    endif()
endif()
