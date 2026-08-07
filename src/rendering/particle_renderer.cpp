#include "particle_renderer.h"

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

layout(location = 0) in vec3 aPosition;

uniform mat4 uView;
uniform mat4 uProjection;

out float vDepth;

void main()
{
    gl_Position = uProjection * uView * vec4(aPosition, 1.0);
    gl_PointSize = 4.0;

    vDepth = clamp(aPosition.y, 0.0, 1.0);
}
)glsl";

constexpr char kParticleFragmentShader[] =
    // language=GLSL
    R"glsl(#version 460 core

in float vDepth;

layout(location = 0) out vec4 fragColor;

void main()
{
    float distanceToCenter = length(gl_PointCoord - vec2(0.5));
    float edgeWidth = fwidth(distanceToCenter);
    float alpha = 1.0 - smoothstep(0.5 - edgeWidth, 0.5, distanceToCenter);

    if (alpha <= 0.0) {
        discard;
    }

    vec3 color = vec3(0.05, 0.35 + 0.65 * vDepth, 1.0);
    fragColor = vec4(color, alpha);
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

ParticleRenderer::ParticleRenderer(std::size_t particleCount)
{
    program_ = createParticleProgram();
    if (program_ == 0) {
        throw std::runtime_error("Failed to create the particle shader program");
    }

    viewLocation_ = glGetUniformLocation(program_, "uView");
    projectionLocation_ = glGetUniformLocation(program_, "uProjection");
    if (viewLocation_ == -1 || projectionLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find shader uniform uView or uProjection");
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

ParticleRenderer::~ParticleRenderer()
{
    shutdown();
}

void ParticleRenderer::render(
    const std::vector<Vec3>& positions,
    const glm::mat4& view,
    const glm::mat4& projection
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

    glProgramUniformMatrix4fv(
        program_,
        viewLocation_,
        1,
        GL_FALSE,
        glm::value_ptr(view)
    );

    glProgramUniformMatrix4fv(
        program_,
        projectionLocation_,
        1,
        GL_FALSE,
        glm::value_ptr(projection)
    );

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);
}

void ParticleRenderer::shutdown()
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
    particleCapacity_ = 0;
}
