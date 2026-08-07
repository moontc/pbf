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

constexpr char kParticleVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core

layout(location = 0) in vec3 aPos;

uniform mat4  uView;
uniform mat4  uProjection;
uniform float uRadius;      // 粒子半径 = d * 0.5
uniform float uViewportH;   // 帧缓冲高度（像素），来自 glfwGetFramebufferSize

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

out vec4 FragColor;

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

    // 用球面法线打光
    vec3  L    = normalize(vec3(0.5, 0.8, 0.6));
    float diff = max(dot(n, L), 0.0);
    FragColor  = vec4(vec3(0.35, 0.65, 0.95) * (0.25 + 0.75 * diff), 1.0);
}
)glsl";

GLuint createParticleProgram()
{
    return linkGlslProgram(
        kParticleVertexShader,
        kParticleFragmentShader,
        "embedded particle shader"
    );
}

} // namespace

FluidRenderer::FluidRenderer(std::size_t particleCount)
{
    program_ = createParticleProgram();
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

    glCreateVertexArrays(1, &vao_);
    glCreateBuffers(1, &vbo_);

    if (vao_ == 0 || vbo_ == 0) {
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

    // Binding 0 reads one packed Vec3 for each vertex.
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

void FluidRenderer::render(
    const std::vector<Vec3>& positions,
    const glm::mat4& view,
    const glm::mat4& projection,
    float radius,
    float viewportHeight
)
{
    if (positions.empty()) {
        return;
    }

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
    glProgramUniform1f(program_, viewportHLocation_, viewportHeight);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);
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

    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }

    viewLocation_ = -1;
    projectionLocation_ = -1;
    viewportHLocation_ = -1;
    radiusLocation_ = -1;
    particleCapacity_ = 0;
}
