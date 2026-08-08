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

// -------------------------- pass 3 ------------------------
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
uniform vec2      uProjXY;       // (proj[0][0], proj[1][1])，见 eyeFromUv

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

    vec3  L    = normalize(vec3(0.5, 0.8, 0.6));
    float diff = max(dot(n, L), 0.0);
    FragColor  = vec4(vec3(0.35, 0.65, 0.95) * (0.25 + 0.75 * diff), 1.0);
}
)glsl";

} // namespace

FluidRenderer::FluidRenderer(std::size_t particleCount)
{
    program_ = linkGlslProgram(
        kParticleVertexShader,
        kParticleFragmentShader,
        "embedded particle shader"
    );
    if (program_ == 0) {
        throw std::runtime_error("Failed to create the particle shader program");
    }

    viewLocation_ = glGetUniformLocation(program_, "uView");
    projectionLocation_ = glGetUniformLocation(program_, "uProjection");
    radiusLocation_ = glGetUniformLocation(program_, "uRadius");
    viewportHLocation_ = glGetUniformLocation(program_, "uViewportH");
    if (viewLocation_ == -1 || projectionLocation_ == -1 ||
        radiusLocation_ == -1 || viewportHLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required particle shader uniform");
    }

    surfaceProgram_ = linkGlslProgram(
        kFullscreenVertexShader,
        kSurfaceFragmentShader,
        "fluid surface shader"
    );
    if (surfaceProgram_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the fluid surface program");
    }

    surfaceProjXYLocation_ = glGetUniformLocation(surfaceProgram_, "uProjXY");
    if (surfaceProjXYLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find shader uniform uProjXY");
    }

    // 告诉 uDepthField 这个 sampler 去 0 号纹理单元取数据。
    //
    // sampler uniform 存的不是纹理，而是一个"纹理单元编号"。运行时用
    // glBindTextureUnit(0, tex) 把具体的纹理挂到 0 号单元上，shader 就读得到。
    // 这个编号存在 program 对象里，设一次就够，不用每帧设。
    glProgramUniform1i(surfaceProgram_, glGetUniformLocation(surfaceProgram_, "uDepthField"), 0);

    glCreateVertexArrays(1, &vao_);
    glCreateVertexArrays(1, &emptyVao_);
    glCreateBuffers(1, &vbo_);

    if (vao_ == 0 || emptyVao_ == 0 || vbo_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the particle VAO or VBO");
    }

    particleCapacity_ = particleCount > 0 ? particleCount : 1;

    glNamedBufferData(
        vbo_,
        static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
        nullptr,
        GL_DYNAMIC_DRAW
    );

    // 绑定点 0 每个顶点读一个紧凑排列的 Vec3。
    glVertexArrayVertexBuffer(
        vao_,
        0,
        vbo_,
        0,
        static_cast<GLsizei>(sizeof(Vec3))
    );
    glEnableVertexArrayAttrib(vao_, 0);
    glVertexArrayAttribFormat(vao_, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao_, 0, 0);
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
    if (depthBuffer_ != 0) { glDeleteRenderbuffers(1, &depthBuffer_); depthBuffer_ = 0; }
    if (fbo_ == 0)         { glCreateFramebuffers(1, &fbo_); }

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

    glCreateRenderbuffers(1, &depthBuffer_);
    glNamedRenderbufferStorage(depthBuffer_, GL_DEPTH_COMPONENT24, width, height);

    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, depthField_, 0);
    glNamedFramebufferRenderbuffer(fbo_, GL_DEPTH_ATTACHMENT,
                                   GL_RENDERBUFFER, depthBuffer_);

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
            vbo_,
            static_cast<GLsizeiptr>(particleCapacity_ * sizeof(Vec3)),
            nullptr,
            GL_DYNAMIC_DRAW
        );
    }

    glNamedBufferSubData(
        vbo_,
        0,
        static_cast<GLsizeiptr>(positions.size() * sizeof(Vec3)),
        positions.data()
    );

    glProgramUniformMatrix4fv(program_, viewLocation_, 1, GL_FALSE, glm::value_ptr(view));
    glProgramUniformMatrix4fv(program_, projectionLocation_, 1, GL_FALSE, glm::value_ptr(projection));
    glProgramUniform1f(program_, radiusLocation_, radius);
    glProgramUniform1f(program_, viewportHLocation_,
                       static_cast<float>(viewportHeight));

    // 绑定fbo
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);

    // -----------------------------------------------------------------------
    // 第二个 pass：读深度场，重建法线打光，画到屏幕
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

    glUseProgram(surfaceProgram_);
    glBindTextureUnit(0, depthField_);   // 把深度场挂到 0 号纹理单元
    glBindVertexArray(emptyVao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);    // 三个顶点，全屏三角形

    glBindVertexArray(0);
    glUseProgram(0);

    // 恢复这个函数改过的两个状态。main.cpp 是在循环外一次性开的深度测试和混合，
    // 这里不恢复，下一帧地板就没有深度测试、也没有径向渐隐了。
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
}

void FluidRenderer::shutdown()
{
    if (vbo_ != 0) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }

    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }

    if (emptyVao_ != 0) {
        glDeleteVertexArrays(1, &emptyVao_);
        emptyVao_ = 0;
    }

    if (depthField_ != 0) {
        glDeleteTextures(1, &depthField_);
        depthField_ = 0;
    }

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

    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
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
