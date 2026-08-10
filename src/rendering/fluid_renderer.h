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

    // pass 2 厚度场：视线在水里走过的总长度，米。Beer-Lambert 的 d 就是它。
    //
    // 和 pass 1 共用同一个顶点着色器（球体 imposter 的几何是一样的），所以这里
    // 也要一套同名的 uniform 位置——它们属于 program 对象，两个 program 各有一份。
    //
    // R16F 够用：值在 0~0.5 m 量级，半精度有约 3 位十进制有效数字，而这个量最终
    // 只是喂给 exp()，不像深度那样要和毫米级窗口做比较。
    GLuint thicknessFbo_ = 0;
    GLuint thicknessField_ = 0;
    GLuint thicknessProgram_ = 0;
    GLint  thickViewLocation_ = -1;
    GLint  thickProjectionLocation_ = -1;
    GLint  thickRadiusLocation_ = -1;
    GLint  thickViewportHLocation_ = -1;

    // pass 3/4 深度平滑
    GLuint blurFbo_ = 0;
    GLuint blurField_ = 0;
    GLuint blurProgram_ = 0;
    GLint  blurDirectionLocation_ = -1;
    GLint  blurRadiusScaleLocation_ = -1;
    GLint  blurScaleLocation_ = -1;
    GLint  blurNarrowRangeLocation_ = -1;

    // pass 5
    GLuint surfaceProgram_ = 0;
    GLuint emptyVao_ = 0;
    GLint  surfaceProjXYLocation_ = -1;
    GLint  surfaceAbsorbLocation_ = -1;

    std::size_t particleCapacity_ = 0;

public:
    // 可调参数，以"粒子半径"为单位
    float blurScale   = 3.0f;   // 滤波半径 = 几倍粒子半径
    float narrowRange = 8.0f;   // 深度窗口 = 几倍粒子半径

    // Beer-Lambert 吸收系数的夸大倍数。
    //
    // shader 里用的是 σ = absorbScale · (0.45, 0.074, 0.02)，括号里是真实水的
    // 吸收系数 (1/m)。absorbScale = 1 就是物理真值，但在 0.5 m 的水体里
    // exp(-0.45·0.5) = 0.80，几乎看不出颜色——真实的水要几米深才明显发蓝。
    // 所以这里刻意放大，用小尺度演示大尺度的现象。
    float absorbScale = 8.0f;
};
