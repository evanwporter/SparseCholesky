cmake_minimum_required(VERSION 3.10)

# Project name and language
project(Sparse LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List of source files for the library
set(LIB_SOURCES
    src/spd.cpp
)

# Create a static library
add_library(libsparse STATIC ${LIB_SOURCES})

# Include directories for the library
target_include_directories(libsparse PUBLIC ${CMAKE_SOURCE_DIR}/include)

# List of source files for the executable
set(EXE_SOURCES
    src/main.cpp
)

# Create the executable
add_executable(sparse ${EXE_SOURCES})

# Link the library into the executable
target_link_libraries(sparse PRIVATE libsparse)
