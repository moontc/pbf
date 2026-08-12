#pragma once

#include <glad/gl.h>
#include <glm/mat4x4.hpp>

inline constexpr const char* kEnvironmentGlsl =
    // language=GLSL
    R"glsl(
// 世界空间的太阳方向
const vec3 kSunDir = normalize(vec3(0.4, 0.75, 0.5));

// dir 为单位世界方向
vec3 environment(vec3 dir) {
    const vec3 kZenith  = vec3(0.10, 0.38, 0.68);  // 天顶，深蓝
    const vec3 kHorizon = vec3(0.72, 0.82, 0.92);  // 地平线附近的白色雾霭
    const vec3 kGround  = vec3(0.76);              // 地平线以下；取 floor.cpp 棋盘 0.753/0.792 的中值，
                                                   // 这样有限大的地板淡出到边缘时不会露出一条色差

    float t = clamp(dir.y, -1.0, 1.0);

    // sqrt 让颜色在地平线附近变化最快，远离地平线后趋于平缓
    vec3 env = t > 0.0
        ? mix(kHorizon, kZenith, sqrt(t))
        : mix(kHorizon, kGround, sqrt(-t));

    // 太阳。指数 900 对应约 2.2° 的半亮角半径，强度 60 使得它在 F₀ = 0.02
    // 的正对入射下反射出来仍然能过曝成白点（60 × 0.02 = 1.2）。
    env += vec3(1.0, 0.97, 0.92) * pow(max(dot(dir, kSunDir), 0.0), 900.0) * 60.0;

    return env;
}
)glsl";

class Sky final {
public:
    Sky();
    ~Sky();

    Sky(const Sky&) = delete;
    Sky& operator=(const Sky&) = delete;

    void render(const glm::mat4& view, const glm::mat4& projection) const;

private:
    void shutdown();

    GLuint program_ = 0;
    GLuint vao_ = 0;

    GLint projXYLocation_ = -1;
    GLint invViewRotLocation_ = -1;
};