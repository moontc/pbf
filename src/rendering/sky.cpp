#include "sky.h"

#include <stdexcept>
#include <string>

#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>

#include "glsl_program.h"

namespace {

constexpr char kVertexShader[] =
    // language=GLSL
        R"glsl(#version 460 core
out vec2 vNdc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vNdc = p * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 0.0, 1.0);
}
)glsl";

constexpr char kFragmentHead[] =
    // language=GLSL
        R"glsl(#version 460 core

uniform vec2 uProjXY;      // (proj[0][0], proj[1][1])
uniform mat3 uInvViewRot;  // 视空间方向 -> 世界方向，即 view 旋转部分的转置

in  vec2 vNdc;
out vec4 FragColor;
)glsl";

constexpr char kFragmentBody[] =
    // language=GLSL
        R"glsl(
void main() {
    // ndc.x = p₀₀ · x / (-z)   =>   x / (-z) = ndc.x / p₀₀
    // 取 z = -1 的那条射线，即 dir_view = (ndc.x / p₀₀, ndc.y / p₁₁, -1)
    vec3 dirView = vec3(vNdc / uProjXY, -1.0);

    FragColor = vec4(environment(normalize(uInvViewRot * dirView)), 1.0);
}
)glsl";

} // namespace

Sky::Sky()
{
    const std::string fragmentSource =
        std::string(kFragmentHead) + kEnvironmentGlsl + kFragmentBody;

    program_ = buildGlslProgram(kVertexShader, fragmentSource.c_str(), "sky shader");
    if (program_ == 0) {
        throw std::runtime_error("Failed to create the sky program");
    }

    projXYLocation_     = glGetUniformLocation(program_, "uProjXY");
    invViewRotLocation_ = glGetUniformLocation(program_, "uInvViewRot");
    if (projXYLocation_ == -1 || invViewRotLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a required sky shader uniform");
    }

    glCreateVertexArrays(1, &vao_);
}

Sky::~Sky()
{
    shutdown();
}

void Sky::shutdown()
{
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

void Sky::render(const glm::mat4& view, const glm::mat4& projection) const
{
    glProgramUniform2f(program_, projXYLocation_, projection[0][0], projection[1][1]);

    const glm::mat3 invViewRot = glm::transpose(glm::mat3(view));
    glProgramUniformMatrix3fv(program_, invViewRotLocation_, 1, GL_FALSE, &invViewRot[0][0]);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glUseProgram(0);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}