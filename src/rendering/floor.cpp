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

uniform vec2  uCenter;      // 模拟盒底面在 xz 平面上的中心
uniform float uCell;        // 棋盘格方块边长，单位为世界空间单位
uniform float uFadeStart;   // 平面开始淡出的半径
uniform float uFadeEnd;     // 平面完全消失的半径

layout(location = 0) out vec4 fragColor;

// 周期为 2、值域为 [0,1] 的三角波；它是生成棋盘格单轴图案的方波的原函数。
float triWave(float t)
{
    return 1.0 - 2.0 * abs(fract(t * 0.5) - 0.5);
}

// 计算该方波（值域为 -1..+1）在区间 [t - w/2, t + w/2] 上的平均值。
// 因为 triWave 是方波的积分，所以只需精确计算两个 triWave 样本之差，
// 不需要采样或循环。
float meanSquare(float t, float w)
{
    return (triWave(t + 0.5 * w) - triWave(t - 0.5 * w)) / w;
}

// 经过方框滤波的棋盘格。
//
// 直接使用 mod(floor(c.x) + floor(c.y), 2.0) 时，只要一个像素覆盖多个方格就会
// 产生摩尔纹；在地面上离开中心几个方格后就会出现。fwidth 无法像处理网格线那样
// 修复它，因为这里没有可单独柔化的一条边缘，而是整个图案都欠采样。
//
// 因此这里采用积分而不是点采样。棋盘格可分解为两个方波的乘积，w 是以方格为单位
// 的像素覆盖范围，每个轴都在该范围内求平均。覆盖范围超过一个方格后，平均值会
// 衰减到 0.5，与多级渐远纹理最终收敛的中间色相同，但这里完全通过解析计算实现，
// 不需要纹理。
float checker(vec2 p, float size)
{
    vec2 c = p / size;
    vec2 w = fwidth(c) + 1e-4;   // 加上 eps，因为 w 将作为除数
    return 0.5 - 0.5 * meanSquare(c.x, w.x) * meanSquare(c.y, w.y);
}

void main()
{
    vec2 p = vWorld.xz;

    const vec3 kLight = vec3(0.792157);   // #CACACA
    const vec3 kDark  = vec3(0.752941);   // #C0C0C0

    vec3 color = mix(kDark, kLight, checker(p, uCell));

    // 使用径向淡出，使该四边形看起来像开放地面，而不是悬浮在空中的矩形平板。
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
