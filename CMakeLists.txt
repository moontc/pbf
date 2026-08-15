cmake_minimum_required(VERSION 4.2)
project(pbf LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual")

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)

FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v2.0.8
        SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(glad)

glad_add_library(
        glad_gl_core_46
        REPRODUCIBLE
        API gl:core=4.6
)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(glm)

find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(pbf_solver STATIC
        src/solvers/cuda_pbf_solver.cu
        src/solvers/cuda_pbf_solver.cuh
        src/solvers/pbf_solver.h
        src/solvers/pbf_types.h
        src/vector.h)

target_include_directories(pbf_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pbf_solver PUBLIC
        OpenMP::OpenMP_CXX
        CUDA::cudart
)

# CUB（随 CCCL 提供）无法使用 MSVC 的传统预处理器构建。
# 该选项只用于 CUDA 源文件，不影响 C++ 源文件。
target_compile_options(pbf_solver PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

add_executable(pbf src/main.cpp
        src/rendering/camera.cpp
        src/rendering/camera.h
        src/rendering/glsl_program.h
        src/rendering/floor.cpp
        src/rendering/floor.h
        src/rendering/sky.cpp
        src/rendering/sky.h
        src/frame_profiler.h
        src/rendering/fluid_renderer.cpp
        src/rendering/fluid_renderer.h)

target_link_libraries(pbf PRIVATE
        pbf_solver
        glfw
        glad_gl_core_46
        glm::glm
        OpenGL::GL
)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt)
    add_subdirectory(test)
endif()
