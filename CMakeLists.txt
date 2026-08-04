cmake_minimum_required(VERSION 4.2)
project(pbf)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(pbf src/main.cpp
        src/camera.cpp
        src/camera.h
        src/vector.h
        src/Shader.cpp
        src/Shader.h
        src/pbf_solver.h)

target_include_directories(pbf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pbf PRIVATE
        glfw
        glad_gl_core_46
        glm::glm
        OpenGL::GL
        OpenMP::OpenMP_CXX
)

add_executable(test test/test.cpp)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
