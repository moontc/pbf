#pragma once

#include <cstddef>
#include <vector>

#include <glad/gl.h>
#include <glm/mat4x4.hpp>

#include "vector.h"

class ParticleRenderer final {
public:
    explicit ParticleRenderer(std::size_t particleCount);
    ~ParticleRenderer();

    ParticleRenderer(const ParticleRenderer&) = delete;
    ParticleRenderer& operator=(const ParticleRenderer&) = delete;

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
