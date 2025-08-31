cmake_minimum_required(VERSION 3.10)

project(Sparse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB_SOURCES
    src/chol.cpp
    src/pprinter.cpp
)

find_package(OpenBLAS CONFIG REQUIRED)
find_package(lapack CONFIG REQUIRED)

add_library(libsparse STATIC ${LIB_SOURCES})

target_include_directories(libsparse 
    PUBLIC ${CMAKE_SOURCE_DIR}/include 
           ${CMAKE_SOURCE_DIR}/pcg-cpp/include)

target_link_libraries(libsparse 
    PUBLIC OpenBLAS::OpenBLAS
           lapack)

set(EXE_SOURCES
    src/main.cpp
)

add_executable(sparse ${EXE_SOURCES})

target_link_libraries(sparse PRIVATE libsparse)
