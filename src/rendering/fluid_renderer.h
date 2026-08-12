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

    void blurThicknessField(const glm::mat4& projection, float radius,
                            int viewportWidth, int viewportHeight);


    void shadeSurface(const glm::mat4& view, const glm::mat4& projection, float radius);

    // 深度场：到流体表面的视空间距离，没有流体的地方是 0
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

    // 厚度场：视线在水里走过的总长度，米
    GLuint thicknessFbo_ = 0;
    GLuint thicknessField_ = 0;
    GLuint thicknessProgram_ = 0;
    GLint  thickViewLocation_ = -1;
    GLint  thickProjectionLocation_ = -1;
    GLint  thickRadiusLocation_ = -1;
    GLint  thickViewportHLocation_ = -1;

    // 可分离高斯，深度场和厚度场共用同一个 program
    GLuint blurFbo_ = 0;
    GLuint blurField_ = 0;
    GLuint blurProgram_ = 0;
    GLint  blurDirectionLocation_ = -1;
    GLint  blurRadiusScaleLocation_ = -1;
    GLint  blurScaleLocation_ = -1;

    // 背景：画水之前的整幅屏幕内容，折射用它当被错位采样的对象。
    // 不挂 FBO，只被 glCopyTextureSubImage2D 写、被表面着色器读。
    GLuint backgroundField_ = 0;

    // 表面着色：读平滑后的深度场、厚度场和背景，重建法线 + 折射/反射/菲涅耳。
    // emptyVao_ 给全屏三角形用——核心 profile 下没绑 VAO 的 glDrawArrays 什么都不画。
    GLuint surfaceProgram_ = 0;
    GLuint emptyVao_ = 0;
    GLint  surfaceProjXYLocation_ = -1;
    GLint  surfaceAbsorbLocation_ = -1;
    GLint  surfaceNormalThresholdLocation_ = -1;
    GLint  surfaceInvViewRotLocation_ = -1;
    GLint  surfaceRefractLocation_ = -1;

    std::size_t particleCapacity_ = 0;

public:
    // 空间标准差 σ_s 对应的滤波半径，以"粒子半径"为单位
    float blurScale   = 3.0f;

    // 法线重建时判定"这两侧是不是同一层水面"的深度差阈值，单位是粒子半径。
    // uNormalDepthThreshold = 2 · sigmaRange · r。
    float sigmaRange  = 4.0f;

    // Beer-Lambert 吸收系数的夸大倍数。
    // σ = absorbScale · (0.45, 0.074, 0.02)
    float absorbScale = 2.0f;

    // 折射错位量的人为倍数
    // Δuv = (1 - 1/η) · thickness · -n.xy · (p₀₀, p₁₁) / (2z)
    float refractScale = 1.0f;
};
