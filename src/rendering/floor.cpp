#include "floor.h"

#include <algorithm>
#include <stdexcept>

#include <glm/gtc/type_ptr.hpp>

#include "glsl_program.h"

namespace {

constexpr char kFloorVertexShader[] =
    // language=GLSL
    R"glsl(#version 460 core

layout(location = 0) in vec3 aPosition;

uniform mat4 uViewProjection;

out vec3 vWorld;

void main()
{
    vWorld = aPosition;
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
}
)glsl";

constexpr char kFloorFragmentShader[] =
    // language=GLSL
    R"glsl(#version 460 core

in vec3 vWorld;

uniform vec2  uCenter;      // xz centre of the simulation box footprint
uniform float uCell;        // checker square size, world units
uniform float uFadeStart;   // radius where the plane starts fading out
uniform float uFadeEnd;     // radius where it is fully gone

layout(location = 0) out vec4 fragColor;

// Triangle wave, period 2, range [0,1].  This is the antiderivative of the
// square wave that generates one axis of the checkerboard.
float triWave(float t)
{
    return 1.0 - 2.0 * abs(fract(t * 0.5) - 0.5);
}

// Mean value of that square wave (which runs -1..+1) over the interval
// [t - w/2, t + w/2].  Because triWave integrates the square wave, the mean is
// an exact difference of two triWave samples -- no sampling, no loop.
float meanSquare(float t, float w)
{
    return (triWave(t + 0.5 * w) - triWave(t - 0.5 * w)) / w;
}

// Box-filtered checkerboard.
//
// The naive form -- mod(floor(c.x) + floor(c.y), 2.0) -- aliases into moire as
// soon as one pixel spans more than one square, which on a ground plane happens
// only a few squares out.  fwidth cannot rescue it the way it rescues a grid
// line: there is no single edge to soften, the whole pattern is under-sampled.
//
// So integrate instead of point-sample.  The checker is separable into a
// product of two square waves, w is the pixel footprint measured in squares,
// and each axis is averaged over exactly that footprint.  Once the footprint
// exceeds one square the average decays to 0.5 -- the same mid tone a mipmap
// would converge to, but computed analytically and with no texture at all.
float checker(vec2 p, float size)
{
    vec2 c = p / size;
    vec2 w = fwidth(c) + 1e-4;   // +eps: w is a divisor
    return 0.5 - 0.5 * meanSquare(c.x, w.x) * meanSquare(c.y, w.y);
}

void main()
{
    vec2 p = vWorld.xz;

    const vec3 kLight = vec3(0.792157);   // #CACACA
    const vec3 kDark  = vec3(0.752941);   // #C0C0C0

    vec3 color = mix(kDark, kLight, checker(p, uCell));

    // Radial fade, so the quad reads as an open ground plane rather than a
    // rectangular slab floating in the void.
    float alpha = 1.0 - smoothstep(uFadeStart, uFadeEnd, length(p - uCenter));

    if (alpha <= 0.0) {
        discard;
    }

    fragColor = vec4(color, alpha);
}
)glsl";

} // namespace

Floor::Floor(const Vec3& boxLo, const Vec3& boxHi)
{
    program_ = buildGlslProgram(
        kFloorVertexShader,
        kFloorFragmentShader,
        "embedded floor shader"
    );
    if (program_ == 0) {
        throw std::runtime_error("Failed to create the floor shader program");
    }

    viewProjectionLocation_ = glGetUniformLocation(program_, "uViewProjection");
    centerLocation_ = glGetUniformLocation(program_, "uCenter");
    cellLocation_ = glGetUniformLocation(program_, "uCell");
    fadeStartLocation_ = glGetUniformLocation(program_, "uFadeStart");
    fadeEndLocation_ = glGetUniformLocation(program_, "uFadeEnd");

    if (viewProjectionLocation_ == -1 ||
        centerLocation_ == -1 ||
        cellLocation_ == -1 ||
        fadeStartLocation_ == -1 ||
        fadeEndLocation_ == -1) {
        shutdown();
        throw std::runtime_error("Cannot find a floor shader uniform");
    }

    centerX_ = 0.5f * (boxLo.x + boxHi.x);
    centerZ_ = 0.5f * (boxLo.z + boxHi.z);

    const float extent = std::max(boxHi.x - boxLo.x, boxHi.z - boxLo.z);

    cell_ = extent / 4.0f;
    fadeStart_ = extent * 4.8f;
    fadeEnd_ = extent * 14.0f;

    const float half = fadeEnd_ * 1.1f;
    const float y = boxLo.y;

    const Vec3 corners[4] = {
        Vec3(centerX_ - half, y, centerZ_ - half),
        Vec3(centerX_ + half, y, centerZ_ - half),
        Vec3(centerX_ - half, y, centerZ_ + half),
        Vec3(centerX_ + half, y, centerZ_ + half),
    };

    glCreateVertexArrays(1, &vao_);
    glCreateBuffers(1, &vbo_);

    if (vao_ == 0 || vbo_ == 0) {
        shutdown();
        throw std::runtime_error("Failed to create the floor VAO or VBO");
    }

    glNamedBufferStorage(vbo_, sizeof(corners), corners, 0);

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

Floor::~Floor()
{
    shutdown();
}

void Floor::render(const glm::mat4& viewProjection) const
{
    glProgramUniformMatrix4fv(
        program_,
        viewProjectionLocation_,
        1,
        GL_FALSE,
        glm::value_ptr(viewProjection)
    );
    glProgramUniform2f(program_, centerLocation_, centerX_, centerZ_);
    glProgramUniform1f(program_, cellLocation_, cell_);
    glProgramUniform1f(program_, fadeStartLocation_, fadeStart_);
    glProgramUniform1f(program_, fadeEndLocation_, fadeEnd_);

    glUseProgram(program_);
    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glUseProgram(0);
}

void Floor::shutdown()
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
    centerLocation_ = -1;
    cellLocation_ = -1;
    fadeStartLocation_ = -1;
    fadeEndLocation_ = -1;
}