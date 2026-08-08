#pragma once

#include <cstddef>
#include <vector>

#include <glad/gl.h>
#include <glm/mat4x4.hpp>

#include "../vector.h"

class FluidRenderer final {
public:
    explicit FluidRenderer(std::size_t particleCount);
    ~FluidRenderer();

    FluidRenderer(const FluidRenderer&) = delete;
    FluidRenderer& operator=(const FluidRenderer&) = delete;

    void render(
        const std::vector<Vec3>& positions,
        const glm::mat4& view,
        const glm::mat4& projection,
        float radius,
        int viewportWidth,
        int viewportHeight
    );

private:
    void shutdown();

    // 尺寸变化时重建离屏目标
    void ensureTarget(int width, int height);

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLint viewLocation_ = -1;
    GLint projectionLocation_ = -1;
    GLint radiusLocation_ = -1;
    GLint viewportHLocation_ = -1;

    // pass 1 深度场：到流体表面的眼空间距离，没有流体的地方是 0。
    GLuint fbo_ = 0;
    GLuint depthField_ = 0;
    GLuint depthBuffer_ = 0;
    int    targetWidth_ = 0;
    int    targetHeight_ = 0;

    // pass 3
    GLuint surfaceProgram_ = 0;
    GLuint emptyVao_ = 0;
    GLint  surfaceProjXYLocation_ = -1;

    std::size_t particleCapacity_ = 0;
};
