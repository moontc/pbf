cmake_minimum_required(VERSION 4.2)
project(pbf LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)

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

# GLM 1.0.1 still declares CMake 3.6 compatibility.  Evaluate the dependency
# with at least CMake 3.10 policies so CMake 4.x does not emit deprecation warnings.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(glm)

find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The solvers, as a library, so that both the application and the validation
# executable share one compilation of the ~40 s CUDA translation unit.
add_library(pbf_solver STATIC
        src/cuda_pbf_solver.cu
        src/cuda_pbf_solver.cuh
        src/pbf_solver.h
        src/vector.h)

target_include_directories(pbf_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pbf_solver PUBLIC
        OpenMP::OpenMP_CXX
        CUDA::cudart
)

# CUB (shipped as CCCL) refuses to build against MSVC's traditional
# preprocessor.  CUDA sources only -- the C++ ones are unaffected.
#
# Deliberately absent: -use_fast_math.  It is the CUDA equivalent of
# -ffast-math, and the same reason applies (see CLAUDE.md): the reassociation
# it permits breaks the exact pairwise cancellation of the position deltas,
# which is what gives momentum conservation.
target_compile_options(pbf_solver PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

# CUDA_SEPARABLE_COMPILATION is deliberately off: there is a single .cu and no
# __device__ function is called across translation units, so -rdc=true would buy
# nothing and cost the whole-program optimisation of the device code.  It also
# keeps this static library free of a device-link step.

# ---------------------------------------------------------------- the app ---
add_executable(pbf src/main.cpp
        src/camera.cpp
        src/camera.h
        src/particle_renderer.cpp
        src/particle_renderer.h
        src/glsl_program.h
        src/floor.cpp
        src/floor.h
        src/frame_profiler.h)

target_link_libraries(pbf PRIVATE
        pbf_solver
        glfw
        glad_gl_core_46
        glm::glm
        OpenGL::GL
)

# --------------------------------------------------------------- the tests ---
# Cross-validates the CUDA solver against the CPU one.  No window, no GL
# context; exits non-zero on failure so it can gate a commit.
add_executable(pbf_validate
        test/validate_main.cpp
        test/solver_validate.cpp
        test/solver_validate.h)

target_link_libraries(pbf_validate PRIVATE pbf_solver)

# Headless solver-only benchmark.  Nsight Compute attaches to this rather than
# to the windowed app -- see the comment at the top of bench_main.cpp.
add_executable(pbf_bench test/bench_main.cpp)
target_link_libraries(pbf_bench PRIVATE pbf_solver)

# Offline parameter calibration -- prints a constants block for PbfParams.
# Standalone, no dependency on src/.
add_executable(params_test test/params_test.cpp)
target_include_directories(params_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
