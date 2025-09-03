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
find_package(Eigen3 CONFIG REQUIRED)

add_library(libsparse STATIC ${LIB_SOURCES})

target_include_directories(libsparse 
    PUBLIC ${CMAKE_SOURCE_DIR}/include 
           ${CMAKE_SOURCE_DIR}/pcg-cpp/include)

target_link_libraries(libsparse 
    PUBLIC OpenBLAS::OpenBLAS
           lapack
           Eigen3::Eigen)

set(EXE_SOURCES
    src/main.cpp
)

add_executable(sparse ${EXE_SOURCES})

target_link_libraries(sparse PRIVATE libsparse)

include(FetchContent)
FetchContent_Declare(
  googletest
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(CTest)
enable_testing()

set(TEST_SOURCES tests/test_chol.cpp)

add_executable(chol_tests ${TEST_SOURCES})
target_link_libraries(chol_tests PRIVATE libsparse gtest_main)

add_test(NAME chol_tests COMMAND chol_tests)
