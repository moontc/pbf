#pragma once

#include <cstddef>
#include <vector>

#include <glad/gl.h>

#include "vector.h"

class Renderer final {
public:
    explicit Renderer(std::size_t particleCount);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(
        const std::vector<Vec3>& positions
    );

private:
    void shutdown();

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;

    std::size_t particleCapacity_ = 0;
};
