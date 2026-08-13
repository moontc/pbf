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
        src/rendering/camera.cpp
        src/rendering/camera.h
        src/rendering/particle_renderer.cpp
        src/rendering/particle_renderer.h
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

# --------------------------------------------------------------- the tests ---
# Local-only tools (cross-validation, benchmark, parameter calibration).  They
# live in test/CMakeLists.txt so that nothing above this line knows about them,
# and the whole directory -- that file included -- is covered by .gitignore's
# /test entry.
#
# The EXISTS guard is the point, not defensive noise: a fresh clone has no test/
# at all, and declaring those targets unconditionally from here made configure
# fail outright on three of them.  Must come after add_library(pbf_solver ...),
# since all three link against it.
#
# Trade-off, accepted deliberately: a fresh clone cannot run pbf_validate, which
# is the only oracle for the CUDA port.  Anyone needing it has to obtain test/
# out of band.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt)
    add_subdirectory(test)
else ()
    message(STATUS "test/ not present -- skipping pbf_validate, pbf_bench, params_test")
endif ()
