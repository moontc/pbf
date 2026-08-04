#include "shader.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <glm/gtc/type_ptr.hpp>

namespace {

static_assert(std::is_standard_layout_v<Vec3>,
              "Vec3 must have a standard memory layout");
static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "Vec3 must contain exactly three packed floats");

constexpr char kParticleVertexShader[] =
    // language=GLSL
    R"glsl(#version 460 core

layout(location = 0) in vec3 aPosition;

uniform mat4 uViewProjection;

out float vDepth;

void main()
{
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
    gl_PointSize = 3.0;

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

    vec3 color = vec3(0.05, 0.35 + 0.55 * vDepth, 1.0);
    fragColor = vec4(color, alpha);
}
)glsl";

GLuint compileShader(
    GLenum type,
    const char* source,
    const char* name
)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_TRUE) {
        return shader;
    }

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

    std::string log(
        static_cast<std::size_t>(logLength > 0 ? logLength : 1),
        '\0'
    );
    glGetShaderInfoLog(shader, logLength, nullptr, log.data());

    const char* typeName = type == GL_VERTEX_SHADER
        ? "vertex shader"
        : "fragment shader";
    std::fprintf(
        stderr,
        "Failed to compile %s '%s':\n%s\n",
        typeName,
        name,
        log.c_str()
    );

    glDeleteShader(shader);
    return 0;
}

GLuint createParticleProgram()
{
    const GLuint vertexShader = compileShader(
        GL_VERTEX_SHADER,
        kParticleVertexShader,
        "embedded particle vertex shader"
    );
    if (vertexShader == 0) {
        return 0;
    }

    const GLuint fragmentShader = compileShader(
        GL_FRAGMENT_SHADER,
        kParticleFragmentShader,
        "embedded particle fragment shader"
    );
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return 0;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_TRUE) {
        return program;
    }

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

    std::string log(
        static_cast<std::size_t>(logLength > 0 ? logLength : 1),
        '\0'
    );
    glGetProgramInfoLog(program, logLength, nullptr, log.data());
    std::fprintf(stderr, "Failed to link particle program:\n%s\n", log.c_str());

    glDeleteProgram(program);
    return 0;
}

} // namespace

shader::shader(std::size_t particleCount)
{
    program_ = createParticleProgram();
    if (program_ == 0) {
        throw std::runtime_error("Failed to create the particle shader program");
    }

    viewProjectionLocation_ = glGetUniformLocation(program_, "uViewProjection");
    if (viewProjectionLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find shader uniform uViewProjection");
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

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

shader::~shader()
{
    shutdown();
}

void shader::render(
    const std::vector<Vec3>& positions,
    const glm::mat4& viewProjection
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
        viewProjectionLocation_,
        1,
        GL_FALSE,
        glm::value_ptr(viewProjection)
    );

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));
    glBindVertexArray(0);
    glUseProgram(0);
}

void shader::shutdown()
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

    viewProjectionLocation_ = -1;
    particleCapacity_ = 0;
}
