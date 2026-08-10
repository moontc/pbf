#include "fluid_renderer.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <glm/gtc/type_ptr.hpp>

#include "glsl_program.h"

namespace {

static_assert(std::is_standard_layout_v<Vec3>,
              "Vec3 must have a standard memory layout");
static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must contain exactly three packed floats");

// ---------------------- pass 1 -------------------------
constexpr char kParticleVertexShader[] =
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

constexpr char kParticleFragmentShader[] =
    // language=GLSL
        R"glsl(#version 460 core
in vec3 vViewPos;

uniform mat4  uProjection;
uniform float uRadius;

layout(location = 0) out float fragDistance;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;                          // gl_PointCoord 的 y 朝下
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;

    vec3 n = vec3(c, sqrt(1.0 - r2));           // 视空间球面法线
    vec3 spherePos = vViewPos + n * uRadius;    // 这个片元真正的 3D 位置

    // 真实球面深度
    vec4 clip = uProjection * vec4(spherePos, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;

    // 眼空间距离，正数。0 留给"这里没有流体"，后面每一个 pass 判断背景都靠它，
    // 所以不能用 gl_FragDepth 那种非线性值——平滑滤波要按米比较深度窗口。
    fragDistance = -spherePos.z;
}
)glsl";
// ----------------------------------------------------------

// -------------------------- pass 2 ------------------------
// 厚度场：视线在水里走过的总长度。顶点着色器和 pass 1 共用。
constexpr char kThicknessFragmentShader[] =
    // language=GLSL
        R"glsl(#version 460 core

uniform float uRadius;

layout(location = 0) out float fragThickness;

void main() {
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float c2 = dot(c, c);
    if (c2 > 1.0) discard;

    // L = 2·r·sqrt(1 - |c|²)
    fragThickness = 2.0 * uRadius * sqrt(1.0 - c2);
}
)glsl";
// ----------------------------------------------------------

// -------------------------- pass 3/4 ------------------------
// 深度平滑：窄范围滤波 (narrow-range filter, Truong & Yuksel 2018)
constexpr char kBlurFragmentShader[] =
    // language=GLSL
        R"glsl(#version 460 core

uniform sampler2D uSource;
uniform vec2      uDirection;     // 一个 texel 的步长：(1/w,0) 或 (0,1/h)
uniform float     uRadiusScale;   // 粒子在 z=1m 处的像素半径
uniform float     uBlurScale;     // 滤波半径 = 几倍粒子半径
uniform float     uNarrowRange;   // 深度窗口，米

layout(location = 0) out float fragDistance;

// 循环上界必须是编译期常量，实际半径靠 break 截断
const int kMaxRadius = 32;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uSource, 0));

    float z0 = texture(uSource, uv).r;
    if (z0 <= 0.0) {
        fragDistance = 0.0;
        return;
    }

    // R = uBlurScale · uRadiusScale / z,   uRadiusScale = viewportH·proj[1][1]·r / 2
    int R = int(min(uBlurScale * uRadiusScale / z0, float(kMaxRadius)));
    if (R < 1) {
        fragDistance = z0;
        return;
    }

    // 高斯权重  w(i) = exp(-i² / (2σ²))，取 σ = R/2。
    // 最外圈 i=R 处 w = exp(-R²/(2·(R/2)²)) = exp(-2) ≈ 0.135，截断不可见。
    float sigma  = float(R) * 0.5;
    float inv2s2 = 1.0 / (2.0 * sigma * sigma);

    float sum     = z0;      // i=0 的中心样本，w(0) = exp(0) = 1
    float weights = 1.0;

    for (int i = 1; i <= kMaxRadius; ++i) {
        if (i > R) break;

        float w = exp(-float(i * i) * inv2s2);

        for (int side = -1; side <= 1; side += 2) {   // 对称的两侧
            float zi = texture(uSource, uv + uDirection * float(i * side)).r;

            if (zi <= 0.0) continue;
            if (zi < z0 - uNarrowRange) continue;

            sum     += w * min(zi, z0 + uNarrowRange);
            weights += w;
        }
    }

    fragDistance = sum / weights;
}
)glsl";
// ----------------------------------------------------------

// -------------------------- pass 4 ------------------------
constexpr char kFullscreenVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

// 从深度场重建表面法线，然后打光
constexpr char kSurfaceFragmentShader[] =
    // language=GLSL
        R"glsl(#version 460 core

uniform sampler2D uDepthField;   // R32F，眼空间距离，0 表示这里没有流体
uniform sampler2D uThickness;    // R16F，视线穿过的水的总长度，米
uniform vec2      uProjXY;       // (proj[0][0], proj[1][1])，见 eyeFromUv
uniform vec3      uAbsorb;       // Beer-Lambert 吸收系数 σ，1/m，分通道

out vec4 FragColor;

float depthAt(vec2 uv) {
    return texture(uDepthField, uv).r;
}

// ndc.x = proj[0][0] * eye.x / (-eye.z)
// eye.x = ndc.x * z / proj[0][0]
vec3 eyeFromUv(vec2 uv, float z) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * z / uProjXY.x,
                ndc.y * z / uProjXY.y,
                -z);
}

vec3 tangent(vec2 uv, vec2 step, vec3 here, float z0) {
    float zp = depthAt(uv + step);
    float zm = depthAt(uv - step);

    if (zp <= 0.0 && zm <= 0.0) return vec3(0.0);
    if (zm <= 0.0) return eyeFromUv(uv + step, zp) - here;
    if (zp <= 0.0) return here - eyeFromUv(uv - step, zm);

    return abs(zp - z0) < abs(zm - z0)
        ? eyeFromUv(uv + step, zp) - here
        : here - eyeFromUv(uv - step, zm);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uDepthField, 0));
    vec2 uv    = gl_FragCoord.xy * texel;

    float z = depthAt(uv);
    if (z <= 0.0) discard;

    vec3 here = eyeFromUv(uv, z);

    vec3 tx = tangent(uv, vec2(texel.x, 0.0), here, z);
    vec3 ty = tangent(uv, vec2(0.0, texel.y), here, z);

    vec3 n = cross(tx, ty);
    n = dot(n, n) > 1e-18 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    if (n.z < 0.0) n = -n;

    // Beer-Lambert：光穿过吸收介质，强度按路径长度指数衰减
    //     I(d) = I₀ · exp(-σ · d)
    // σ 分通道，水对红光的吸收远强于蓝光——**这才是水呈蓝绿色的原因**，不是因为
    // 它反射蓝光。所以这里不再乘任何人为的蓝色调，蓝绿是算出来的：薄处
    // exp(0) = 1 出射接近白色，厚处红光先被吃掉，剩下青蓝。
    float thickness = texture(uThickness, uv).r;
    vec3  transmit  = exp(-uAbsorb * thickness);

    vec3  L    = normalize(vec3(0.5, 0.8, 0.6));
    float diff = max(dot(n, L), 0.0);
    FragColor  = vec4(transmit * (0.25 + 0.75 * diff), 1.0);
}
)glsl";

} // namespace

FluidRenderer::FluidRenderer(std::size_t particleCount)
{
    depthProgram_ = buildGlslProgram(
        kParticleVertexShader,
        kParticleFragmentShader,
        "embedded particle shader");
    if (depthProgram_ == 0) {
        throw std::runtime_error("Failed to create the particle shader program");
    }

    viewLocation_ = glGetUniformLocation(depthProgram_, "uView");
    projectionLocation_ = glGetUniformLocation(depthProgram_, "uProjection");
    radiusLocation_ = glGetUniformLocation(depthProgram_, "uRadius");
    viewportHLocation_ = glGetUniformLocation(depthProgram_, "uViewportH");
    if (viewLocation_ == -1 || projectionLocation_ == -1 ||
        radiusLocation_ == -1 || viewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required particle shader uniform");
    }

    thicknessProgram_ = buildGlslProgram(
        kParticleVertexShader,
        kThicknessFragmentShader,
        "fluid thickness shader"
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

    blurProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        kBlurFragmentShader,
        "fluid depth smoothing"
    );
    if (blurProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the depth smoothing program");
    }

    blurDirectionLocation_   = glGetUniformLocation(blurProgram_, "uDirection");
    blurRadiusScaleLocation_ = glGetUniformLocation(blurProgram_, "uRadiusScale");
    blurScaleLocation_       = glGetUniformLocation(blurProgram_, "uBlurScale");
    blurNarrowRangeLocation_ = glGetUniformLocation(blurProgram_, "uNarrowRange");
    if (blurDirectionLocation_ == -1 || blurRadiusScaleLocation_ == -1 ||
        blurScaleLocation_ == -1 || blurNarrowRangeLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required blur shader uniform");
    }

    glProgramUniform1i(blurProgram_, glGetUniformLocation(blurProgram_, "uSource"), 0);

    surfaceProgram_ = buildGlslProgram(
        kFullscreenVertexShader,
        kSurfaceFragmentShader,
        "fluid surface shader"
    );
    if (surfaceProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid surface program");
    }

    surfaceProjXYLocation_ = glGetUniformLocation(surfaceProgram_, "uProjXY");
    surfaceAbsorbLocation_ = glGetUniformLocation(surfaceProgram_, "uAbsorb");
    if (surfaceProjXYLocation_ == -1 || surfaceAbsorbLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required surface shader uniform");
    }

    // 两个 sampler 分别去 0 号和 1 号纹理单元取数据
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uDepthField"), 0);
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uThickness"), 1);

    glCreateVertexArrays(1, &depthVao_);
    glCreateVertexArrays(1, &emptyVao_);
    glCreateBuffers(1, &depthVbo_);

    if (depthVao_ == 0 || emptyVao_ == 0 || depthVbo_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the particle VAO or VBO");
    }

    particleCapacity_ = particleCount > 0 ? particleCount : 1;

    glNamedBufferData(
        depthVbo_,
        static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
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

    // glTextureStorage2D 创建的是不可变纹理，所以尺寸变化只能删掉重建，
    // 不能重新分配。FBO 本身可以复用，只换附件。
    if (depthField_ != 0)  { glDeleteTextures(1, &depthField_); depthField_ = 0; }
    if (blurField_ != 0)   { glDeleteTextures(1, &blurField_); blurField_ = 0; }
    if (thicknessField_ != 0) { glDeleteTextures(1, &thicknessField_); thicknessField_ = 0; }
    if (thicknessFbo_ == 0)   { glCreateFramebuffers(1, &thicknessFbo_); }
    if (depthBuffer_ != 0) { glDeleteRenderbuffers(1, &depthBuffer_); depthBuffer_ = 0; }
    if (fbo_ == 0)         { glCreateFramebuffers(1, &fbo_); }
    if (blurFbo_ == 0)     { glCreateFramebuffers(1, &blurFbo_); }

    targetWidth_  = width;
    targetHeight_ = height;

    glCreateTextures(GL_TEXTURE_2D, 1, &depthField_);
    glTextureStorage2D(depthField_, 1, GL_R32F, width, height);
    // NEAREST：后面的 pass 只在精确的 texel 中心采样，插值不但没有意义，
    // 还会跨轮廓把前后两团流体的深度混在一起。
    // CLAMP_TO_EDGE：滤波会采到图像外面，包裹模式会把屏幕另一边折进表面。
    glTextureParameteri(depthField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(depthField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 平滑的中转纹理，格式和采样方式必须与 depthField_ 完全一致——横向那一遍的
    // 输出就是竖向那一遍的输入，两者精度不同会让第二遍在第一遍的量化台阶上工作。
    glCreateTextures(GL_TEXTURE_2D, 1, &blurField_);
    glTextureStorage2D(blurField_, 1, GL_R32F, width, height);
    glTextureParameteri(blurField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(blurField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateRenderbuffers(1, &depthBuffer_);
    glNamedRenderbufferStorage(depthBuffer_, GL_DEPTH_COMPONENT24, width, height);

    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, depthField_, 0);
    glNamedFramebufferRenderbuffer(fbo_, GL_DEPTH_ATTACHMENT,
                                   GL_RENDERBUFFER, depthBuffer_);

    // 厚度场。同样不需要深度附件——这个 pass 的深度测试本来就是关掉的，那正是
    // 它能把前表面后面的球也累加进来的原因。
    glCreateTextures(GL_TEXTURE_2D, 1, &thicknessField_);
    glTextureStorage2D(thicknessField_, 1, GL_R16F, width, height);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(thicknessField_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glNamedFramebufferTexture(thicknessFbo_, GL_COLOR_ATTACHMENT0, thicknessField_, 0);

    if (glCheckNamedFramebufferStatus(thicknessFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid thickness framebuffer is incomplete");
    }

    // blurFbo_ 不需要深度附件：平滑是全屏 pass，深度测试是关掉的。
    glNamedFramebufferTexture(blurFbo_, GL_COLOR_ATTACHMENT0, blurField_, 0);

    if (glCheckNamedFramebufferStatus(blurFbo_, GL_FRAMEBUFFER)
            != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid blur framebuffer is incomplete");
    }

    const GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Fluid framebuffer is incomplete: 0x"
                                 + std::to_string(status));
    }
}

void FluidRenderer::render(
    const std::vector<Vec3>& positions,
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    int viewportWidth,
    int viewportHeight
)
{
    if (positions.empty() || viewportWidth <= 0 || viewportHeight <= 0) {
        return;
    }

    ensureTarget(viewportWidth, viewportHeight);

    if (positions.size() > particleCapacity_) {
        particleCapacity_ = positions.size();
        glNamedBufferData(
            depthVbo_,
            static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
            nullptr,
            GL_DYNAMIC_DRAW
        );
    }

    glNamedBufferSubData(
        depthVbo_,
        0,
        static_cast<GLsizeiptr>(positions.size() * sizeof(Vec3)),
        positions.data()
    );

    glProgramUniformMatrix4fv(depthProgram_, viewLocation_, 1, GL_FALSE, glm::value_ptr(view));
    glProgramUniformMatrix4fv(depthProgram_, projectionLocation_, 1, GL_FALSE, glm::value_ptr(projection));
    glProgramUniform1f(depthProgram_, radiusLocation_, radius);
    glProgramUniform1f(depthProgram_, viewportHLocation_,
                       static_cast<float>(viewportHeight));

    // 绑定fbo
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(depthProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);

    // -----------------------------------------------------------------------
    // pass 2：厚度场
    // -----------------------------------------------------------------------
    // 状态和 pass 1 正好相反：
    //   深度测试关 —— 故意让前表面后面的球也画上，累加的就是总路径长度
    //   加性混合   —— GL_ONE, GL_ONE 就是 dst = dst + src，所以弦长会累加
    // 清成 0 是加性混合的前提，否则累加会从上一帧的残留开始。
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
    glBlendFunc(GL_ONE, GL_ONE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(thicknessProgram_);
    glBindVertexArray(depthVao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);

    // 混合模式改回来，后面的 pass 和 main.cpp 的地板都按 SRC_ALPHA 混合。
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // -----------------------------------------------------------------------
    // pass 3/4：深度平滑，横竖各一遍
    // -----------------------------------------------------------------------
    // 深度测试关掉，全屏 pass 不需要它；顺手把深度写也关掉，免得往 fbo_ 的深度
    // renderbuffer 里写进无意义的 0.5。
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    // uRadiusScale = viewportH · proj[1][1] · r / 2
    // 也就是这颗粒子在 z = 1 m 处占的像素半径。shader 里再除以逐像素的 z。
    const float radiusScale = static_cast<float>(viewportHeight) * projection[1][1] * radius * 0.5f;

    glProgramUniform1f(blurProgram_, blurRadiusScaleLocation_, radiusScale);
    glProgramUniform1f(blurProgram_, blurScaleLocation_, blurScale);
    // narrowRange 在头文件里以"粒子半径"为单位，这里换成米交给 shader。
    glProgramUniform1f(blurProgram_, blurNarrowRangeLocation_, narrowRange * radius);

    glUseProgram(blurProgram_);
    glBindVertexArray(emptyVao_);

    // 横向：读 depthField_ -> 写 blurField_
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo_);
    glBindTextureUnit(0, depthField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       1.0f / static_cast<float>(viewportWidth), 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // 竖向：读 blurField_ -> 写回 depthField_
    // 读和写是两张不同的纹理，所以没有反馈；两遍之后 depthField_ 就是平滑结果。
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glBindTextureUnit(0, blurField_);
    glProgramUniform2f(blurProgram_, blurDirectionLocation_,
                       0.0f, 1.0f / static_cast<float>(viewportHeight));
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glDepthMask(GL_TRUE);

    // -----------------------------------------------------------------------
    // pass 5：读深度场和厚度场，重建法线 + Beer-Lambert，画到屏幕
    // -----------------------------------------------------------------------
    // 绑回 0 号帧缓冲，也就是窗口本身。不解绑的话，下一帧 main.cpp 的清屏和地板
    // 会画进 FBO 里，屏幕就再也不刷新了（症状是画面卡在第一帧）。
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 深度测试要关。全屏三角形的 gl_Position.z 写死是 0，换算成深度值是 0.5，
    // 这个数和地板谁近谁远纯属偶然——开着深度测试会随机剔掉一部分像素。
    // 挡不挡地板是靠片元里的 discard 决定的，不靠深度。
    glDisable(GL_DEPTH_TEST);

    // eyeFromUv 反投影只需要这两个矩阵元素。glm 是列主序，projection[列][行]，
    // 所以 [0][0] 和 [1][1] 就是对角线上的那两项。
    glProgramUniform2f(surfaceProgram_, surfaceProjXYLocation_,
                       projection[0][0], projection[1][1]);

    // σ = absorbScale · (0.45, 0.074, 0.02)，括号里是真实水的吸收系数 (1/m)。
    // 红光被吸收得比蓝光强 22 倍，这就是水呈蓝绿色的来源。
    glProgramUniform3f(surfaceProgram_, surfaceAbsorbLocation_,
                       absorbScale * 0.45f,
                       absorbScale * 0.074f,
                       absorbScale * 0.02f);

    glUseProgram(surfaceProgram_);
    glBindTextureUnit(0, depthField_);       // 0 号纹理单元 = uDepthField
    glBindTextureUnit(1, thicknessField_);   // 1 号纹理单元 = uThickness
    glBindVertexArray(emptyVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);    // 三个顶点，全屏三角形

    glBindVertexArray(0);
    glUseProgram(0);

    // 恢复函数改过的两个状态
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
}

void FluidRenderer::shutdown()
{
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
    blurNarrowRangeLocation_ = -1;

    if (depthBuffer_ != 0) {
        glDeleteRenderbuffers(1, &depthBuffer_);
        depthBuffer_ = 0;
    }

    if (fbo_ != 0) {
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
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

    surfaceProjXYLocation_ = -1;

    viewLocation_ = -1;
    projectionLocation_ = -1;
    viewportHLocation_ = -1;
    radiusLocation_ = -1;
    particleCapacity_ = 0;
}
