#include "fluid_renderer.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_gl_interop.h>
#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "glsl_program.h"
#include "sky.h"
#include "../solvers/cuda_pbf_solver.cuh"

namespace {

static_assert(std::is_standard_layout_v<Vec3>,
              "Vec3 must have a standard memory layout");
static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must contain exactly three packed floats");

void checkCuda(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": "
                             + cudaGetErrorString(error));
}

constexpr char kImposterVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core

layout(location = 0) in vec3 aPos;

uniform mat4  uView;
uniform mat4  uProjection;
uniform float uRadius;
uniform float uViewportH;   // 帧缓冲高度

out vec3 vViewPos;

void main() {
    vec4 view = uView * vec4(aPos, 1.0);
    vViewPos = view.xyz;

    gl_Position = uProjection * view;

    gl_PointSize = uViewportH * uProjection[1][1] * uRadius / -view.z;
}
)glsl";

// 深度平滑、厚度平滑和表面着色共用
constexpr char kFullscreenVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

} // namespace

FluidRenderer::FluidRenderer(std::size_t particleCount)
{
    initDepthPass();
    initThicknessPass();
    initBlurPass();
    initSurfacePass();

    glCreateVertexArrays(1, &depthVao_);
    glCreateVertexArrays(1, &emptyVao_);
    glCreateBuffers(1, &depthVbo_);

    if (depthVao_ == 0 || emptyVao_ == 0 || depthVbo_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the particle VAO or VBO");
    }

    particleCount_ = particleCount;
    const std::size_t allocationCount = particleCount_ > 0 ? particleCount_ : 1;

    glNamedBufferData(
        depthVbo_,
        static_cast<GLsizeiptr>(allocationCount * sizeof(Vec3)),
        nullptr,
        GL_DYNAMIC_DRAW
    );

    // 绑定点 0 每个顶点读一个紧凑排列的 Vec3。
    glVertexArrayVertexBuffer(
        depthVao_,
        0,
        depthVbo_,
        0,
        static_cast<GLsizei>(sizeof(Vec3))
    );
    glEnableVertexArrayAttrib(depthVao_, 0);
    glVertexArrayAttribFormat(depthVao_, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(depthVao_, 0, 0);

    try {
        checkCuda(cudaGraphicsGLRegisterBuffer(
                      &cudaPositions_, depthVbo_,
                      cudaGraphicsRegisterFlagsWriteDiscard),
                  "cudaGraphicsGLRegisterBuffer failed");
    } catch (...) {
        shutdown();
        throw;
    }
}

FluidRenderer::~FluidRenderer()
{
    shutdown();
}

void FluidRenderer::ensureTarget(int width, int height)
{
    if (width == targetWidth_ && height == targetHeight_) {
        return;
    }

    // glTextureStorage2D 创建的是不可变纹理，尺寸变化应重建
    if (depthField_ != 0)  { glDeleteTextures(1, &depthField_); depthField_ = 0; }
    if (blurField_ != 0)   { glDeleteTextures(1, &blurField_); blurField_ = 0; }
    if (thicknessField_ != 0) { glDeleteTextures(1, &thicknessField_); thicknessField_ = 0; }
    if (backgroundField_ != 0) { glDeleteTextures(1, &backgroundField_); backgroundField_ = 0; }
    if (depthBuffer_ != 0) { glDeleteRenderbuffers(1, &depthBuffer_); depthBuffer_ = 0; }

    if (depthFbo_ == 0)         { glCreateFramebuffers(1, &depthFbo_); }
    if (blurFbo_ == 0)     { glCreateFramebuffers(1, &blurFbo_); }
    if (thicknessFbo_ == 0)   { glCreateFramebuffers(1, &thicknessFbo_); }

    targetWidth_  = width;
    targetHeight_ = height;

    glCreateTextures(GL_TEXTURE_2D, 1, &depthField_);
    glTextureStorage2D(depthField_, 1, GL_R32F, width, height);

    // NEAREST：后面的 pass 只在精确的 texel 中心采样，插值没有意义，还会跨轮廓把前后两团流体的深度混在一起
    // CLAMP_TO_EDGE：滤波会采到图像外面，包裹模式会把屏幕另一边折进表面
    glTextureParameteri(depthField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 平滑的中转纹理，格式和采样方式与 depthField_ 一致
    glCreateTextures(GL_TEXTURE_2D, 1, &blurField_);
    glTextureStorage2D(blurField_, 1, GL_R32F, width, height);
    glTextureParameteri(blurField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateRenderbuffers(1, &depthBuffer_);
    glNamedRenderbufferStorage(depthBuffer_, GL_DEPTH_COMPONENT24, width, height);

    glNamedFramebufferTexture(depthFbo_, GL_COLOR_ATTACHMENT0, depthField_, 0);
    glNamedFramebufferRenderbuffer(depthFbo_, GL_DEPTH_ATTACHMENT,
                                   GL_RENDERBUFFER, depthBuffer_);

    // 厚度场
    glCreateTextures(GL_TEXTURE_2D, 1, &thicknessField_);
    glTextureStorage2D(thicknessField_, 1, GL_R16F, width, height);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glNamedFramebufferTexture(thicknessFbo_, GL_COLOR_ATTACHMENT0, thicknessField_, 0);

    // 背景，格式与窗口一致，为 RGBA8
    glCreateTextures(GL_TEXTURE_2D, 1, &backgroundField_);
    glTextureStorage2D(backgroundField_, 1, GL_RGBA8, width, height);
    glTextureParameteri(backgroundField_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(backgroundField_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(backgroundField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(backgroundField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if (glCheckNamedFramebufferStatus(thicknessFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid thickness framebuffer is incomplete");
    }

    // blurFbo_ 不需要深度附件
    glNamedFramebufferTexture(blurFbo_, GL_COLOR_ATTACHMENT0, blurField_, 0);

    if (glCheckNamedFramebufferStatus(blurFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid blur framebuffer is incomplete");
    }

    const GLenum status = glCheckNamedFramebufferStatus(depthFbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid framebuffer is incomplete: 0x"
                                 + std::to_string(status));
    }
}

void FluidRenderer::updatePositions(const CudaPbfSolver& solver)
{
    if (static_cast<std::size_t>(solver.count()) != particleCount_) {
        throw std::runtime_error(
            "CUDA solver particle count does not match the registered OpenGL VBO");
    }
    if (particleCount_ == 0) return;

    checkCuda(cudaGraphicsMapResources(1, &cudaPositions_),
              "cudaGraphicsMapResources failed");

    bool mapped = true;
    try {
        void* devicePointer = nullptr;
        std::size_t mappedBytes = 0;
        checkCuda(cudaGraphicsResourceGetMappedPointer(
                      &devicePointer, &mappedBytes, cudaPositions_),
                  "cudaGraphicsResourceGetMappedPointer failed");

        if (mappedBytes < particleCount_ * sizeof(Vec3)) {
            throw std::runtime_error("CUDA-mapped OpenGL VBO is too small");
        }

        solver.copyPositionsToDevice(static_cast<Vec3*>(devicePointer),
                                     mappedBytes / sizeof(Vec3));

        const cudaError_t unmapError =
            cudaGraphicsUnmapResources(1, &cudaPositions_);
        mapped = false;
        checkCuda(unmapError, "cudaGraphicsUnmapResources failed");
    } catch (...) {
        if (mapped) cudaGraphicsUnmapResources(1, &cudaPositions_);
        throw;
    }
}

void FluidRenderer::updatePositions(const std::vector<Vec3>& positions)
{
    if (positions.size() != particleCount_) {
        throw std::runtime_error(
            "CPU position count does not match the registered OpenGL VBO");
    }
    if (particleCount_ == 0) return;

    glNamedBufferSubData(
        depthVbo_,
        0,
        static_cast<GLsizeiptr>(positions.size() * sizeof(Vec3)),
        positions.data()
    );
}

void FluidRenderer::render(
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    if (particleCount_ == 0 || viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    ensureTarget(viewportWidth, viewportHeight);
    const GLsizei count = static_cast<GLsizei>(particleCount_);

    drawDepthField(view, projection, radius, viewportHeight, count);
    drawThicknessField(view, projection, radius, viewportHeight, count);
    blurDepthField(projection, radius, viewportWidth, viewportHeight);
    blurThicknessField(projection, radius, viewportWidth, viewportHeight);
    shadeSurface(view, projection, radius);

    // 恢复状态
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// 深度场
void FluidRenderer::initDepthPass()
{
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core
in vec3 vViewPos;

uniform mat4  uProjection;
uniform float uRadius;

layout(location = 0) out float fragDistance;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;                                 // gl_PointCoord 的 y 朝下
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;

    vec3 n = vec3(c, sqrt(1.0 - r2));           // 视空间球面法线
    vec3 spherePos = vViewPos + n * uRadius;    // 这个片元真正的 3D 位置

    // 真实球面深度
    vec4 clip = uProjection * vec4(spherePos, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;

    fragDistance = -spherePos.z;
}
)glsl";

    depthProgram_ = buildGlslProgram(
        kImposterVertexShader,
        kFragmentShader,
        "fluid depth field"
    );
    if (depthProgram_ == 0) {
        throw std::runtime_error("Failed to create the fluid depth program");
    }

    viewLocation_       = glGetUniformLocation(depthProgram_, "uView");
    projectionLocation_ = glGetUniformLocation(depthProgram_, "uProjection");
    radiusLocation_     = glGetUniformLocation(depthProgram_, "uRadius");
    viewportHLocation_  = glGetUniformLocation(depthProgram_, "uViewportH");
    if (viewLocation_ == -1 || projectionLocation_ == -1 ||
        radiusLocation_ == -1 || viewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required depth shader uniform");
    }
}

void FluidRenderer::drawDepthField(
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportHeight,
    GLsizei count
)
{
    glProgramUniformMatrix4fv(depthProgram_, viewLocation_, 1, GL_FALSE,
                              glm::value_ptr(view));
    glProgramUniformMatrix4fv(depthProgram_, projectionLocation_, 1, GL_FALSE,
                              glm::value_ptr(projection));
    glProgramUniform1f(depthProgram_, radiusLocation_, radius);
    glProgramUniform1f(depthProgram_, viewportHLocation_, static_cast<float>(viewportHeight));

    glBindFramebuffer(GL_FRAMEBUFFER, depthFbo_);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(depthProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glUseProgram(0);
}

// 厚度场
void FluidRenderer::initThicknessPass()
{
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform float uRadius;

layout(location = 0) out float fragThickness;

void main() {
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float c2 = dot(c, c);
    if (c2 > 1.0) discard;

    // 视线穿过一颗球的弦长  L = 2·r·sqrt(1 - |c|²)
    fragThickness = 2.0 * uRadius * sqrt(1.0 - c2);
}
)glsl";

    thicknessProgram_ = buildGlslProgram(
        kImposterVertexShader,
        kFragmentShader,
        "fluid thickness field"
    );
    if (thicknessProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid thickness program");
    }

    thickViewLocation_       = glGetUniformLocation(thicknessProgram_, "uView");
    thickProjectionLocation_ = glGetUniformLocation(thicknessProgram_, "uProjection");
    thickRadiusLocation_     = glGetUniformLocation(thicknessProgram_, "uRadius");
    thickViewportHLocation_  = glGetUniformLocation(thicknessProgram_, "uViewportH");
    if (thickViewLocation_ == -1 || thickProjectionLocation_ == -1 ||
        thickRadiusLocation_ == -1 || thickViewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required thickness shader uniform");
    }
}

// 视线穿过水体的总长度，单位米
void FluidRenderer::drawThicknessField(
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportHeight,
    GLsizei count
)
{
    glProgramUniformMatrix4fv(thicknessProgram_, thickViewLocation_, 1, GL_FALSE,
                              glm::value_ptr(view));
    glProgramUniformMatrix4fv(thicknessProgram_, thickProjectionLocation_, 1, GL_FALSE,
                              glm::value_ptr(projection));
    glProgramUniform1f(thicknessProgram_, thickRadiusLocation_, radius);
    glProgramUniform1f(thicknessProgram_, thickViewportHLocation_,
                       static_cast<float>(viewportHeight));

    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFbo_);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    // 加性混合：GL_ONE，每颗球的弦长相加
    glBlendFunc(GL_ONE, GL_ONE);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(thicknessProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    glUseProgram(0);
}

// 平滑
void FluidRenderer::initBlurPass()
{
    constexpr char kFragmentShader[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform sampler2D uSource;        // 要平滑的场：深度场或厚度场
uniform sampler2D uDepth;         // 深度场；平滑深度时它和 uSource 是同一张纹理
uniform vec2      uDirection;     // 一个 texel 的步长：(1/w,0) 或 (0,1/h)
uniform float     uRadiusScale;   // 粒子在 z=1m 处的像素半径
uniform float     uBlurScale;     // 滤波半径 = 几倍粒子半径

layout(location = 0) out float fragValue;

// 循环上界必须是编译期常量，实际半径靠 break 截断。
const int kMaxRadius = 32;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uSource, 0));
    vec2 uv    = gl_FragCoord.xy * texel;

    float z0 = texture(uDepth, uv).r;
    if (z0 <= 0.0) {
        fragValue = 0.0;
        return;
    }

    float v0 = texture(uSource, uv).r;   // 中心的待平滑值

    // R = uBlurScale · uRadiusScale / z，最大不超过 kMaxRadius。
    int R = int(min(uBlurScale * uRadiusScale / z0, float(kMaxRadius)));
    if (R < 1) {
        fragValue = v0;
        return;
    }

    // 高斯权重  w(i) = exp(-i² / (2σ²))，取 σ = R/2。
    // 最外圈 i=R 处 w = exp(-R²/(2·(R/2)²)) = exp(-2) ≈ 0.135，截断不可见。
    float sigma  = float(R) * 0.5;
    float inv2s2 = 1.0 / (2.0 * sigma * sigma);

    float sum     = v0;      // i=0 的中心样本，w(0) = exp(0) = 1
    float weights = 1.0;

    for (int i = 1; i <= kMaxRadius; ++i) {
        if (i > R) break;

        float w = exp(-float(i * i) * inv2s2);

        for (int side = -1; side <= 1; side += 2) {   // 对称的两侧
            vec2  suv = uv + uDirection * float(i * side);
            float zi  = texture(uDepth, suv).r;

            // 空白区域不参与平均
            if (zi <= 0.0) continue;

            sum     += w * texture(uSource, suv).r;
            weights += w;
        }
    }

    fragValue = sum / weights;
}
)glsl";

    blurProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        kFragmentShader,
        "fluid depth smoothing"
    );
    if (blurProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the depth smoothing program");
    }

    blurDirectionLocation_   = glGetUniformLocation(blurProgram_, "uDirection");
    blurRadiusScaleLocation_ = glGetUniformLocation(blurProgram_, "uRadiusScale");
    blurScaleLocation_       = glGetUniformLocation(blurProgram_, "uBlurScale");
    if (blurDirectionLocation_ == -1 || blurRadiusScaleLocation_ == -1 ||
        blurScaleLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required blur shader uniform");
    }

    // 给 shader 中的两个 sampler2D 指定固定的纹理单元
    glProgramUniform1i(blurProgram_, glGetUniformLocation(blurProgram_, "uSource"), 0);
    glProgramUniform1i(blurProgram_, glGetUniformLocation(blurProgram_, "uDepth"), 1);
}

// 深度平滑
void FluidRenderer::blurDepthField(
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    // 全屏 pass，不需要深度测试
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);

    // uRadiusScale = viewportH · proj[1][1] · r / 2
    // 也就是这颗粒子在 z = 1 m 处占的像素半径。shader 里再除以逐像素的 z，
    // 于是远处的流体用小窗口、近处用大窗口，滤波强度在视觉上保持一致。
    const float radiusScale =
        static_cast<float>(viewportHeight) * projection[1][1] * radius * 0.5f;

    glProgramUniform1f(blurProgram_, blurRadiusScaleLocation_, radiusScale);
    glProgramUniform1f(blurProgram_, blurScaleLocation_, blurScale_);

    glUseProgram(blurProgram_);
    glBindVertexArray(emptyVao_);

    // 横向：depthField_ -> blurField_
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo_);
    glBindTextureUnit(0, depthField_);
    glBindTextureUnit(1, depthField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       1.0f / static_cast<float>(viewportWidth), 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // 竖向：blurField_ -> depthField_
    glBindFramebuffer(GL_FRAMEBUFFER, depthFbo_);
    glBindTextureUnit(0, blurField_);
    glBindTextureUnit(1, blurField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       0.0f, 1.0f / static_cast<float>(viewportHeight));
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
}

// 厚度平滑
void FluidRenderer::blurThicknessField(
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);

    const float radiusScale =
        static_cast<float>(viewportHeight) * projection[1][1] * radius * 0.5f;

    glProgramUniform1f(blurProgram_, blurRadiusScaleLocation_, radiusScale);
    glProgramUniform1f(blurProgram_, blurScaleLocation_, blurScale_);

    glUseProgram(blurProgram_);
    glBindVertexArray(emptyVao_);

    glBindTextureUnit(1, depthField_);

    // thicknessField_ -> blurField_
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo_);
    glBindTextureUnit(0, thicknessField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       1.0f / static_cast<float>(viewportWidth), 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // blurField_ -> thicknessField_
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFbo_);
    glBindTextureUnit(0, blurField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       0.0f, 1.0f / static_cast<float>(viewportHeight));
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
}

// 表面着色
void FluidRenderer::initSurfacePass()
{
    constexpr char kFragmentHead[] =
        // language=GLSL
            R"glsl(#version 460 core

uniform sampler2D uDepthField;   // R32F，视空间距离，0 表示这里没有流体
uniform sampler2D uThickness;    // R16F，视线穿过的水的总长度，米
uniform sampler2D uBackground;   // 画水之前的屏幕内容
uniform vec2      uProjXY;       // (proj[0][0], proj[1][1])
uniform vec3      uAbsorb;       // Beer-Lambert 吸收系数 σ，1/m，分通道
uniform float     uNormalDepthThreshold; // 中央差分允许跨过的最大深度差，米
uniform mat3      uInvViewRot;   // 视空间方向 -> 世界方向，即 view 旋转部分的转置
uniform float     uRefract;      // 折射错位的人为倍数，1 = 按下面推导出的物理值

out vec4 FragColor;

// 空气 -> 水的折射率
const float kEta = 1.333;

// 垂直入射时的菲涅耳反射率
// F₀ = ((η₁ - η₂) / (η₁ + η₂))² = ((1 - 1.333) / (1 + 1.333))² ≈ 0.02
const float kF0 = 0.02;

float depthAt(vec2 uv) {
    return texture(uDepthField, uv).r;
}
)glsl";

    constexpr char kFragmentBody[] =
        // language=GLSL
            R"glsl(
// ndc.x = proj[0][0] * view.x / (-view.z)
// view.x = ndc.x * z / proj[0][0]
vec3 viewFromUv(vec2 uv, float z) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * z / uProjXY.x,
                ndc.y * z / uProjXY.y,
                -z);
}

vec3 tangent(vec2 uv, vec2 step, vec3 here, float z0) {
    float zp = depthAt(uv + step);
    float zm = depthAt(uv - step);

    if (zp <= 0.0 && zm <= 0.0) return vec3(0.0);
    if (zm <= 0.0) return viewFromUv(uv + step, zp) - here;
    if (zp <= 0.0) return here - viewFromUv(uv - step, zm);

    float deltaPlus  = abs(zp - z0);
    float deltaMinus = abs(zm - z0);
    float span       = abs(zp - zm);

    if (span <= uNormalDepthThreshold) {
        return 0.5 * (viewFromUv(uv + step, zp)
                    - viewFromUv(uv - step, zm));
    }

    return deltaPlus < deltaMinus
        ? viewFromUv(uv + step, zp) - here
        : here - viewFromUv(uv - step, zm);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uDepthField, 0));
    vec2 uv    = gl_FragCoord.xy * texel;

    float z = depthAt(uv);
    if (z <= 0.0) discard;

    vec3 here = viewFromUv(uv, z);

    vec3 tx = tangent(uv, vec2(texel.x, 0.0), here, z);
    vec3 ty = tangent(uv, vec2(0.0, texel.y), here, z);

    vec3 n = cross(tx, ty);
    n = dot(n, n) > 1e-18 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    if (n.z < 0.0) n = -n;

    float thickness = texture(uThickness, uv).r;

    // 视线方向
    vec3  v        = normalize(here);
    float cosTheta = clamp(dot(n, -v), 0.0, 1.0);

    // 折射：错位采样背景
    // 屏幕空间小角度近似：把视线近似为视空间 Z 轴，则单位法线 n 的
    // |n.xy| = sinθ，方向也由 n.xy 给出。由 Snell 定律近似折射横移，再将其投影为
    // UV 偏移：Δuv ≈ (1 - 1/η) · thickness · n.xy · (p₀₀,p₁₁) / (2z)。
    vec2 duv = (1.0 - 1.0 / kEta) * thickness * -n.xy * uProjXY / (2.0 * z);

    // 夹住偏移量：轮廓上法线几乎垂直于视线，Δuv 会大到把屏幕另一头的东西拉过来。
    duv = clamp(duv * uRefract, vec2(-0.15), vec2(0.15));

    vec3 behind = texture(uBackground, clamp(uv + duv, vec2(0.0), vec2(1.0))).rgb;

    // Beer-Lambert：光穿过吸收介质，强度按路径长度指数衰减
    // I(d) = I₀ · exp(-σ · d)
    // σ 为三个通道的吸收系数
    vec3 refracted = behind * exp(-uAbsorb * thickness);

    // 反射
    // reflect(v, n) = v - 2·dot(v, n)·n，v 射入、n 朝外，得到镜面方向。
    vec3 reflected = environment(uInvViewRot * reflect(v, n));

    // 菲涅耳
    // Schlick 近似  F(θ) = F₀ + (1 - F₀)(1 - cosθ)⁵
    float fresnel = kF0 + (1.0 - kF0) * pow(1.0 - cosTheta, 5.0);

    FragColor = vec4(mix(refracted, reflected, fresnel), 1.0);
}
)glsl";

    const std::string fragmentSource =
        std::string(kFragmentHead) + kEnvironmentGlsl + kFragmentBody;

    surfaceProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        fragmentSource.c_str(),
        "fluid surface shader"
    );
    if (surfaceProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid surface program");
    }

    surfaceProjXYLocation_ = glGetUniformLocation(surfaceProgram_, "uProjXY");
    surfaceAbsorbLocation_ = glGetUniformLocation(surfaceProgram_, "uAbsorb");
    surfaceNormalThresholdLocation_ =
        glGetUniformLocation(surfaceProgram_, "uNormalDepthThreshold");
    surfaceInvViewRotLocation_ = glGetUniformLocation(surfaceProgram_, "uInvViewRot");
    surfaceRefractLocation_    = glGetUniformLocation(surfaceProgram_, "uRefract");
    if (surfaceProjXYLocation_ == -1 || surfaceAbsorbLocation_ == -1 ||
        surfaceNormalThresholdLocation_ == -1 ||
        surfaceInvViewRotLocation_ == -1 || surfaceRefractLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required surface shader uniform");
    }

    // 三个 sampler 分别去 0、1、2 号纹理单元取数据
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uDepthField"), 0);
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uThickness"), 1);
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uBackground"), 2);
}

// 读深度场和厚度场，重建法线 + 折射/反射/菲涅耳，画到屏幕
void FluidRenderer::shadeSurface(const glm::mat4& view, const glm::mat4& projection, float radius)
{
    // 绑回 0 号帧缓冲，也就是窗口本身
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // glCopyTextureSubImage2D 的源是当前绑定的 GL_READ_FRAMEBUFFER
    // glBindFramebuffer(GL_FRAMEBUFFER, 0) 同时设置了 read 和 draw
    glCopyTextureSubImage2D(backgroundField_, 0, 0, 0, 0, 0,
                            targetWidth_, targetHeight_);

    // 关闭深度测试，全屏三角形的 gl_Position.z 恒为 0，与地板的遮挡关系靠片元里的 discard 判定
    // 关闭混合：背景现在由折射项自己采样，输出的 alpha 恒为 1
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // viewFromUv 反投影只需要这两个矩阵元素。glm 是列主序，projection[列][行]，
    glProgramUniform2f(surfaceProgram_, surfaceProjXYLocation_,
                       projection[0][0], projection[1][1]);

    const glm::mat3 invViewRot = glm::transpose(glm::mat3(view));
    glProgramUniformMatrix3fv(surfaceProgram_, surfaceInvViewRotLocation_,
                              1, GL_FALSE, &invViewRot[0][0]);

    glProgramUniform1f(surfaceProgram_, surfaceRefractLocation_, refractScale_);

    // 两侧深度差不超过两个范围标准差时，把它们视为同一层连续表面并使用中央差分
    glProgramUniform1f(surfaceProgram_, surfaceNormalThresholdLocation_,
                       2.0f * sigmaRange_ * radius);

    // σ = absorbScale_ · (0.45, 0.074, 0.02)
    glProgramUniform3f(surfaceProgram_, surfaceAbsorbLocation_,
                       absorbScale_ * 0.45f,
                       absorbScale_ * 0.074f,
                       absorbScale_ * 0.02f);

    glUseProgram(surfaceProgram_);
    glBindTextureUnit(0, depthField_);        // 0 号纹理单元 = uDepthField
    glBindTextureUnit(1, thicknessField_);    // 1 号纹理单元 = uThickness
    glBindTextureUnit(2, backgroundField_);   // 2 号纹理单元 = uBackground
    glBindVertexArray(emptyVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);        // 三个顶点，全屏三角形

    glBindVertexArray(0);
    glUseProgram(0);
}

void FluidRenderer::shutdown()
{
    if (cudaPositions_ != nullptr) {
        const cudaError_t error = cudaGraphicsUnregisterResource(cudaPositions_);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "cudaGraphicsUnregisterResource failed: %s\n",
                         cudaGetErrorString(error));
        }
        cudaPositions_ = nullptr;
    }

    if (depthVbo_ != 0) {
        glDeleteBuffers(1, &depthVbo_);
        depthVbo_ = 0;
    }

    if (depthVao_ != 0) {
        glDeleteVertexArrays(1, &depthVao_);
        depthVao_ = 0;
    }

    if (emptyVao_ != 0) {
        glDeleteVertexArrays(1, &emptyVao_);
        emptyVao_ = 0;
    }

    if (depthField_ != 0) {
        glDeleteTextures(1, &depthField_);
        depthField_ = 0;
    }

    if (blurField_ != 0) {
        glDeleteTextures(1, &blurField_);
        blurField_ = 0;
    }

    if (thicknessField_ != 0) {
        glDeleteTextures(1, &thicknessField_);
        thicknessField_ = 0;
    }

    if (backgroundField_ != 0) {
        glDeleteTextures(1, &backgroundField_);
        backgroundField_ = 0;
    }

    if (thicknessFbo_ != 0) {
        glDeleteFramebuffers(1, &thicknessFbo_);
        thicknessFbo_ = 0;
    }

    if (thicknessProgram_ != 0) {
        glDeleteProgram(thicknessProgram_);
        thicknessProgram_ = 0;
    }

    thickViewLocation_       = -1;
    thickProjectionLocation_ = -1;
    thickRadiusLocation_     = -1;
    thickViewportHLocation_  = -1;
    surfaceAbsorbLocation_   = -1;

    if (blurFbo_ != 0) {
        glDeleteFramebuffers(1, &blurFbo_);
        blurFbo_ = 0;
    }

    if (blurProgram_ != 0) {
        glDeleteProgram(blurProgram_);
        blurProgram_ = 0;
    }

    blurDirectionLocation_   = -1;
    blurRadiusScaleLocation_ = -1;
    blurScaleLocation_       = -1;

    if (depthBuffer_ != 0) {
        glDeleteRenderbuffers(1, &depthBuffer_);
        depthBuffer_ = 0;
    }

    if (depthFbo_ != 0) {
        glDeleteFramebuffers(1, &depthFbo_);
        depthFbo_ = 0;
    }

    targetWidth_ = 0;
    targetHeight_ = 0;

    if (depthProgram_ != 0) {
        glDeleteProgram(depthProgram_);
        depthProgram_ = 0;
    }

    if (surfaceProgram_ != 0) {
        glDeleteProgram(surfaceProgram_);
        surfaceProgram_ = 0;
    }

    surfaceProjXYLocation_          = -1;
    surfaceInvViewRotLocation_      = -1;
    surfaceRefractLocation_         = -1;
    surfaceNormalThresholdLocation_ = -1;

    viewLocation_ = -1;
    projectionLocation_ = -1;
    viewportHLocation_ = -1;
    radiusLocation_ = -1;
    particleCount_ = 0;
}
