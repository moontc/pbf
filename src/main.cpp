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

#include "camera.h"
#include "particle_renderer.h"
#include "floor.h"
#include "frame_profiler.h"
#include "vector.h"
#include "pbf_solver.h"
#include "cuda_pbf_solver.cuh"

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "fail to init GLFW\n");
        return 1;
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
        std::fprintf(stderr, "fail to creat window\n");
        glfwTerminate();

        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::fprintf(stderr, "fail to load GL\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::printf(
        "OpenGL: %s\n",
        reinterpret_cast<const char*>(glGetString(GL_VERSION))
    );

    std::printf("Vendor:   %s\n", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    std::printf("Renderer: %s\n", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    std::printf("Version:  %s\n", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    int device = 0;
    cudaError_t error = cudaGetDevice(&device);

    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "cudaGetDevice failed: %s\n",
            cudaGetErrorString(error)
        );
        return 1;
    }

    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, device);

    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "cudaGetDeviceProperties failed: %s\n",
            cudaGetErrorString(error)
        );
        return 1;
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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    constexpr bool kProfileFrames = true;



    const Vec3 blockLo(0.05f, 0.05f, 0.05f);
    const Vec3 blockHi(0.45f, 0.90f, 0.45f);

    PbfParams params;

    //PbfSolver solver(params);
    CudaPbfSolver solver(params);

    solver.initBlock(blockLo, blockHi);

    int result = 0;

    try {
        Camera camera({0.5, 0.5, 0.25}, 3.0f);
        camera.attach(window);

        ParticleRenderer particles(solver.count());
        Floor floor(params.boxLo, params.boxHi);

        const float FIXED_DT  = 1.0f / 60.0f;
        const int   MAX_STEPS = 5;
        float  accumulator = 0.0f;
        double lastFrameTime = glfwGetTime();

        FrameProfiler profiler(kProfileFrames);

        while (!glfwWindowShouldClose(window)) {

            glfwPollEvents();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            const double currentTime = glfwGetTime();
            float real_dt = static_cast<float>(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            real_dt = std::min(real_dt, 0.025f);

            accumulator += real_dt;

            profiler.frameStart();

            int steps = 0;
            while (accumulator >= FIXED_DT && steps < MAX_STEPS) {
                solver.step(FIXED_DT);
                accumulator -= FIXED_DT;
                ++steps;
            }
            if (steps == MAX_STEPS) accumulator = 0.0f;

            profiler.afterSolve(steps);

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            glViewport(0, 0, width, height);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            const glm::mat4 viewProjection = camera.projection() * camera.view();

            floor.render(viewProjection);
            particles.render(solver.positions(), viewProjection);

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
