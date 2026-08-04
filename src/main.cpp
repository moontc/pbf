#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <vector>
#include <string>

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"
#include "shader.h"
#include "vector.h"
#include "pbf_solver.h"

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

    PbfParams params;
    PbfSolver solver(params);
    solver.initBlock({0.05f, 0.05f, 0.05f}, {0.45f, 0.90f, 0.45f});

    int result = 0;

    try {
        Camera camera({0.5, 0.5, 0.25}, 3.0f);
        camera.attach(window);

        shader shader(solver.count());

        double lastFrameTime = glfwGetTime();

        //TODO FIXED_DT
        while (!glfwWindowShouldClose(window)) {

            glfwPollEvents();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            const double currentTime = glfwGetTime();
            const float dt = static_cast<float>(currentTime - lastFrameTime);
            lastFrameTime = currentTime;

            solver.step(dt);
            auto status = solver.stats();
            printf("fps: %f rhoAvg: %f rhoMax: %f vMax: %f momentum: %f clamped: %d cflHits: %d\n",
                1.0 / dt, status.rhoAvg, status.rhoMax, status.vMax, status.momentum, status.clamped, status.cflHits);

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            glViewport(0, 0, width, height);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            const glm::mat4 viewProjection = camera.projection() * camera.view();

            shader.render(solver.positions(), viewProjection);

            glfwSwapBuffers(window);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return result;
}
