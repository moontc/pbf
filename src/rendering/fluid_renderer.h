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

    void uploadPositions(const std::vector<Vec3>& positions);

    void initDepthPass();
    void initThicknessPass();
    void initBlurPass();
    void initSurfacePass();

    void drawDepthField(const glm::mat4& view, const glm::mat4& projection,
                        float radius, int viewportHeight, GLsizei count);

    void drawThicknessField(const glm::mat4& view, const glm::mat4& projection,
                            float radius, int viewportHeight, GLsizei count);

    void blurDepthField(const glm::mat4& projection, float radius,
                        int viewportWidth, int viewportHeight);

    // 必须排在 blurDepthField 之后：它的滤波半径要用已经平滑好的深度。
    void blurThicknessField(const glm::mat4& projection, float radius,
                            int viewportWidth, int viewportHeight);

    void shadeSurface(const glm::mat4& projection, float radius);

    // 深度场：到流体表面的视空间距离，没有流体的地方是 0。
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

    // 厚度场：视线在水里走过的总长度，米。Beer-Lambert 的 d 就是它。
    GLuint thicknessFbo_ = 0;
    GLuint thicknessField_ = 0;
    GLuint thicknessProgram_ = 0;
    GLint  thickViewLocation_ = -1;
    GLint  thickProjectionLocation_ = -1;
    GLint  thickRadiusLocation_ = -1;
    GLint  thickViewportHLocation_ = -1;

    // 可分离高斯，深度场和厚度场都是横竖各一遍，共用同一个 program。
    GLuint blurFbo_ = 0;
    GLuint blurField_ = 0;
    GLuint blurProgram_ = 0;
    GLint  blurDirectionLocation_ = -1;
    GLint  blurRadiusScaleLocation_ = -1;
    GLint  blurScaleLocation_ = -1;

    // 表面着色：读平滑后的深度场和厚度场，重建法线 + Beer-Lambert，画到屏幕。
    // emptyVao_ 给全屏三角形用——核心 profile 下没绑 VAO 的 glDrawArrays 什么都不画。
    GLuint surfaceProgram_ = 0;
    GLuint emptyVao_ = 0;
    GLint  surfaceProjXYLocation_ = -1;
    GLint  surfaceAbsorbLocation_ = -1;
    GLint  surfaceNormalThresholdLocation_ = -1;

    std::size_t particleCapacity_ = 0;

public:
    // 可调参数，以"粒子半径"为单位
    float blurScale   = 3.0f;   // 空间标准差 σ_s 对应的滤波半径 = 几倍粒子半径

    // 法线重建时判定"这两侧是不是同一层水面"的深度差阈值，单位是粒子半径。
    // shadeSurface 传给 shader 的是 2 · sigmaRange · r。
    //
    // 曾经它还兼任平滑的范围标准差 σ_r（双边 / 窄范围滤波），现在不是了：平滑退回
    // 了纯高斯，权重里不含任何依赖 z 的项。原因见 initBlurPass 顶部的注释。
    float sigmaRange  = 4.0f;

    // Beer-Lambert 吸收系数的夸大倍数。
    //
    // shader 里用的是 σ = absorbScale · (0.45, 0.074, 0.02)，括号里是真实水的
    // 吸收系数 (1/m)。absorbScale = 1 就是物理真值，但在 0.5 m 的水体里
    // exp(-0.45·0.5) = 0.80，几乎看不出颜色——真实的水要几米深才明显发蓝。
    // 所以这里刻意放大，用小尺度演示大尺度的现象。
    float absorbScale = 8.0f;
};
