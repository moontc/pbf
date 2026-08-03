#pragma once

#include <cstddef>
#include <vector>

#include <glad/gl.h>
#include <glm/mat4x4.hpp>

#include "vector.h"

class shader final {
public:
    explicit shader(std::size_t particleCount);
    ~shader();

    shader(const shader&) = delete;
    shader& operator=(const shader&) = delete;

    void render(
        const std::vector<Vec3>& positions,
        const glm::mat4& viewProjection
    );

private:
    void shutdown();

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLint viewProjectionLocation_ = -1;

    std::size_t particleCapacity_ = 0;
};
