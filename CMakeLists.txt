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

# GLM 1.0.1 仍声明兼容 CMake 3.6。至少按 CMake 3.10 的策略处理该依赖，
# 以免 CMake 4.x 产生弃用警告。
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
FetchContent_MakeAvailable(glm)

find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# 将求解器构建为库，使应用程序和验证程序共享同一份编译约需 40 秒的 CUDA 翻译单元。
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
#
# 有意不启用 -use_fast_math。它相当于 CUDA 版本的 -ffast-math；同样地
#（参见 CLAUDE.md），它允许的运算重结合会破坏位置修正量的精确成对抵消，
# 而这种抵消正是动量守恒的基础。
target_compile_options(pbf_solver PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

# 有意关闭 CUDA_SEPARABLE_COMPILATION：项目只有一个 .cu 文件，也没有跨翻译单元
# 调用的 __device__ 函数，因此 -rdc=true 不会带来收益，反而会失去设备代码的
# 全程序优化；同时也能避免为该静态库增加设备链接步骤。

# ---------------------------------------------------------------- 应用程序 ---
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

# --------------------------------------------------------------- 测试工具 ---
# 仅供本地使用的工具（交叉验证、基准测试和参数标定）。它们定义在
# test/CMakeLists.txt 中，所以上面的内容完全不依赖它们；整个目录（包括该文件）
# 都由 .gitignore 中的 /test 规则忽略。
#
# EXISTS 判断是必要设计，而不是多余的防御代码：新克隆的仓库没有 test/ 目录，
# 如果在这里无条件声明这些目标，配置会直接失败。此处必须位于
# add_library(pbf_solver ...) 之后，因为三个目标都链接该库。
#
# 已明确接受的取舍：新克隆的仓库无法运行 pbf_validate，而它是 CUDA 移植版本的
# 唯一正确性基准；需要它的开发者必须通过仓库之外的方式获取 test/ 目录。
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt)
    add_subdirectory(test)
else ()
    message(STATUS "未找到 test/，跳过 pbf_validate、pbf_bench 和 params_test")
endif ()
