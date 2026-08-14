#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;

class Camera
{
public:
    Camera() = default;
    Camera(const glm::vec3& center, float dist);

    glm::mat4 view()           const;
    glm::mat4 projection()     const;

    glm::mat4 invProjection() const;
    glm::mat4 invView()       const;

    void attach(GLFWwindow* window);

private:
    glm::vec3 target_{0.0f};
    float distance_ = 5.0f;
    float yaw_      = -45.0f;
    float pitch_    = 20.0f;

    float fov_    = 45.0f;
    float aspect_ = 16.0f / 9.0f;
    float znear_  = 0.1f;
    float zfar_   = 100.0f;

    float rotateSpeed_ = 0.3f;
    float zoomSpeed_   = 0.1f;
    float minDistance_ = 1.0f;
    float maxDistance_ = 10.0f;
    float maxPitch_    = 89.0f;

    enum class Mode { None, Rotate };

    glm::vec3 position() const;
    void rotate(float dxPixels, float dyPixels);
    void zoom(float steps);

    void onMouseButton(int button, int action, int mods);
    void onCursorPos(double x, double y);
    void onScroll(double dx, double dy);
    void onResize(int width, int height);

    Mode  mode_      = Mode::None;
    bool  firstMove_ = true;
    float lastX_     = 0.0f;
    float lastY_     = 0.0f;
};