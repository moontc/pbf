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
find_package(OpenGL REQUIRED)

add_executable(pbf src/main.cpp
        src/vector.h)
target_include_directories(pbf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pbf PRIVATE
        glfw
        OpenGL::GL
)

add_executable(test test/test.cpp)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
