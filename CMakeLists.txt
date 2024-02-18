cmake_minimum_required(VERSION 2.8)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -g -msse2 -mavx -mfma -Wno-return-type")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17  -Wno-return-type -fopenmp")
#set(CMAKE_CXX_FLAGS "-O3 -std=c++17  -Wno-return-type -DEIGEN_DONT_VECTORIZE")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)
Find_PACKAGE(OpenCV 4 REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system thread)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core g2o_solver_csparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
#        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
#
#target_link_libraries(bundle_adjustment_g2o  boost_thread pthread ${G2O_LIBS} bal_common)
target_link_libraries(bundle_adjustment_g2o  boost_thread pthread ${G2O_LIBS}
        bal_common ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(directBA directBA.cpp
        globalPBA_view.cpp
        globalPBA.cpp)
target_link_libraries(directBA ${G2O_LIBS} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
