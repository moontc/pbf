#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>

#include <type_traits>
#include <vector>
#include <string>

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "rendering/camera.h"
#include "rendering/particle_renderer.h"
#include "rendering/floor.h"
#include "rendering/sky.h"
#include "frame_profiler.h"
#include "vector.h"
#include "solvers/pbf_solver.h"
#include "solvers/cuda_pbf_solver.cuh"
#include "rendering/fluid_renderer.h"

#if defined(_WIN32)
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

GLFWwindow* initOpenGL() {
    if (!glfwInit()) {
        std::fprintf(stderr, "fail to init GLFW\n");
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );
    glfwWindowHint(GLFW_DEPTH_BITS, 24);

    GLFWwindow* window = glfwCreateWindow(
        900,
        900,
        "PBF Particle Simulation",
        nullptr,
        nullptr
    );

    if (!window) {
        std::fprintf(stderr, "fail to create window\n");
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "fail to load GL\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    std::printf(
        "OpenGL: %s\n",
        reinterpret_cast<const char*>(glGetString(GL_VERSION))
    );

    std::printf("Vendor:   %s\n", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    std::printf("Renderer: %s\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    std::printf("Version:  %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    return window;
}

bool initCUDA() {
    unsigned int deviceCount = 0;
    int devices[8]{};
    cudaError_t error = cudaGLGetDevices(&deviceCount, devices, 8,
                                         cudaGLDeviceListAll);

    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "cudaGLGetDevices failed: %s\n",
            cudaGetErrorString(error)
        );
        return false;
    }
    if (deviceCount == 0) {
        std::fprintf(stderr, "OpenGL context is not associated with a CUDA device\n");
        return false;
    }

    const int device = devices[0];
    error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n",
                     cudaGetErrorString(error));
        return false;
    }

    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, device);

    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "cudaGetDeviceProperties failed: %s\n",
            cudaGetErrorString(error)
        );
        return false;
    }

    std::printf("CUDA device: %d\n", device);
    std::printf("CUDA GPU:    %s\n", properties.name);
    std::printf(
        "Compute capability: %d.%d\n",
        properties.major,
        properties.minor
    );
    std::printf(
        "Global memory: %.2f GB\n",
        static_cast<double>(properties.totalGlobalMem)
            / (1024.0 * 1024.0 * 1024.0)
    );
    std::printf(
        "SM count: %d\n",
        properties.multiProcessorCount
    );

    return true;
}

} // namespace

int main() {
    GLFWwindow* window = initOpenGL();
    if (!window) {
        return 1;
    }

    if (!initCUDA()) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    constexpr bool kProfileFrames = true;

    PbfParams params;

    int result = 0;

    try {
        CudaPbfSolver solver(params);

        Camera camera({0.0, 0.5, 0.25}, 4.5f);
        camera.attach(window);

        ParticleRenderer particles(solver.count());
        FluidRenderer fluid(solver.count());
        fluid.updatePositions(solver);
        Floor floor(params.boxLo, params.boxHi);
        Sky sky;

        const float FIXED_DT  = 1.0f / 60.0f;
        float  accumulator = 0.0f;
        double lastFrameTime = glfwGetTime();

        FrameProfiler profiler(kProfileFrames);

        while (!glfwWindowShouldClose(window)) {

            glfwPollEvents();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            const double currentTime = glfwGetTime();
            accumulator += static_cast<float>(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            int steps = 0;
            profiler.frameStart();

            if (accumulator >= FIXED_DT) {
                solver.step(FIXED_DT);
                fluid.updatePositions(solver);
                accumulator = std::fmod(accumulator, FIXED_DT);
                steps = 1;
            }
            cudaDeviceSynchronize();
            profiler.afterSolve(steps);

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            glViewport(0, 0, width, height);

            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int fbW, fbH;
            glfwGetFramebufferSize(window, &fbW, &fbH);

            sky.render(camera.view(), camera.projection());
            floor.render(camera.projection() * camera.view());
            //particles.render(solver.positions(), camera.view(), camera.projection());
            fluid.render(camera.view(), camera.projection(),
                         solver.params().d * 0.7f, fbW, fbH);

            profiler.afterRender();

            glfwSwapBuffers(window);

            profiler.afterSwap(solver.count());
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return result;
}
