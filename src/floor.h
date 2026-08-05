#pragma once

#include <glad/gl.h>
#include <glm/mat4x4.hpp>

#include "vector.h"

class Floor final {
public:
    Floor(const Vec3& boxLo, const Vec3& boxHi);
    ~Floor();

    Floor(const Floor&) = delete;
    Floor& operator=(const Floor&) = delete;

    void render(const glm::mat4& viewProjection) const;

private:
    void shutdown();

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;

    GLint viewProjectionLocation_ = -1;
    GLint centerLocation_ = -1;
    GLint cellLocation_ = -1;
    GLint fadeStartLocation_ = -1;
    GLint fadeEndLocation_ = -1;

    float centerX_ = 0.0f;
    float centerZ_ = 0.0f;

    float cell_ = 0.1f;
    float fadeStart_ = 1.0f;
    float fadeEnd_ = 3.0f;
};