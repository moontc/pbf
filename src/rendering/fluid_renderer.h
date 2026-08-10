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

    // pass 1 深度场：到流体表面的眼空间距离，没有流体的地方是 0。
    GLuint depthProgram_ = 0;
    GLuint depthVao_ = 0;
    GLuint depthVbo_ = 0;
    GLint viewLocation_ = -1;
    GLint projectionLocation_ = -1;
    GLint radiusLocation_ = -1;
    GLint viewportHLocation_ = -1;

    GLuint fbo_ = 0;
    GLuint depthField_ = 0;
    GLuint depthBuffer_ = 0;
    int    targetWidth_ = 0;
    int    targetHeight_ = 0;

    // pass 2/3 深度平滑
    GLuint blurFbo_ = 0;
    GLuint blurField_ = 0;
    GLuint blurProgram_ = 0;
    GLint  blurDirectionLocation_ = -1;
    GLint  blurRadiusScaleLocation_ = -1;
    GLint  blurScaleLocation_ = -1;
    GLint  blurNarrowRangeLocation_ = -1;

    // pass 4
    GLuint surfaceProgram_ = 0;
    GLuint emptyVao_ = 0;
    GLint  surfaceProjXYLocation_ = -1;

    std::size_t particleCapacity_ = 0;

public:
    // 可调参数，以"粒子半径"为单位
    float blurScale   = 3.0f;   // 滤波半径 = 几倍粒子半径
    float narrowRange = 8.0f;   // 深度窗口 = 几倍粒子半径
};
