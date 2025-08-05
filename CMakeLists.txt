cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2 LANGUAGES CXX CUDA)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
add_definitions(-w)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_STANDARD 17)
# ##############################################################################
# Build type and C++ compiler setup
# ##############################################################################

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# ##############################################################################
# Set Python
# ##############################################################################
set(ENVIRONMENT_DIR your_anconda3_path/anaconda3/envs/gsorbslam)
set(PYTHON_INCLUDE_DIR "${ENVIRONMENT_DIR}/include/python3.10")
set(PYTHON_LIB "${ENVIRONMENT_DIR}/lib/libpython3.10.so")


set(Torch_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch/share/cmake/Torch)

# ##############################################################################
# Packages
# ##############################################################################

find_package(OpenCV 4.9 REQUIRED)
if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
else()
   message(STATUS "Opencv Version: ${OpenCV_VERSION}")
   message(STATUS "Opencv Dir: ${OpenCV_DIR}")
endif()


find_package(Eigen3 3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
   message(FATAL_ERROR "TORCH NOT FOUND!")
else()
   message(STATUS "TORCH FOUND!")
   message("Torch_DIR: ${Torch_DIR} ")
   message("Torch_LIB: ${TORCH_LIBRARIES} ")
endif()


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIR}
${Torch_DIR}/../../../include
${PYTHON_NUMPY_INCLUDE}
${PROJECT_SOURCE_DIR}/src/imgui
)

# ##############################################################################
# CUDA compiler setup
# ##############################################################################
set(CUDA_NVCC_FLAGS -gencode arch=compute_89,code=sm_89;-G;-g)
set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++17")
# ##############################################################################
# Program
# ##############################################################################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Render.cc
src/Rasterizer.cu
src/Camera.cc
src/Gaussian.cc
src/Utils.cc
src/spatial.cu
src/simple_knn.cu
src/Viewer2.cc
src/imgui/
)


add_library(imgui SHARED
    src/imgui/imconfig.h
    src/imgui/imgui_demo.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_impl_glfw.cpp
    src/imgui/imgui_impl_glfw.h
    src/imgui/imgui_impl_opengl3_loader.h
    src/imgui/imgui_impl_opengl3.cpp
    src/imgui/imgui_impl_opengl3.h
    src/imgui/imgui_internal.h
    src/imgui/imgui_tables.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui.cpp
    src/imgui/imgui.h
    src/imgui/imstb_rectpack.h
    src/imgui/imstb_textedit.h
    src/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBRARIES}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/diff_gaussian_rasterization/lib/libCudaRasterizer.so
${PYTHON_LIB}
${PROJECT_SOURCE_DIR}/Thirdparty/libtorch/lib/libtorch_python.so
${TORCH_LIBRARIES}
tinyply
${PROJECT_SOURCE_DIR}/Thirdparty/yaml-cpp/build/libyaml-cpp.a
imgui
stdc++fs

# ${PROJECT_SOURCE_DIR}/Thirdparty/wandb-cpp/build/libwandbcpp.so
)

# ##############################################################################
# Build examples
# ##############################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_compile_options(rgbd_tum PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(rgbd_tum ${PROJECT_NAME} )


add_executable(rgbd_replica
Examples/RGB-D/rgbd_replica.cc)
target_compile_options(rgbd_replica PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(rgbd_replica ${PROJECT_NAME} )

add_executable(rgbd_scannet
Examples/RGB-D/rgbd_scannet.cc)
target_compile_options(rgbd_scannet PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(rgbd_scannet ${PROJECT_NAME} )

# /usr/local/lib/libtiff.so.5
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

# add_executable(stereo_kitti
# Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})

# add_executable(stereo_euroc
# Examples/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc ${PROJECT_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

# add_executable(mono_tum
# Examples/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum ${PROJECT_NAME})

# add_executable(mono_kitti
# Examples/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti ${PROJECT_NAME})

# add_executable(mono_euroc
# Examples/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc ${PROJECT_NAME})

