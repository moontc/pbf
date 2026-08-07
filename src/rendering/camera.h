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
    glm::vec3 target{0.0f};
    float distance = 5.0f;
    float yaw      = -90.0f;
    float pitch    = 20.0f;

    float fov    = 45.0f;
    float aspect = 16.0f / 9.0f;
    float znear  = 0.1f;
    float zfar   = 100.0f;

    float rotateSpeed = 0.3f;
    float zoomSpeed   = 0.1f;
    float minDistance = 0.05f;
    float maxDistance = 1000.0f;
    float maxPitch    = 89.0f;

    enum class Mode { None, Rotate };

    glm::vec3 position() const;
    void rotate(float dxPixels, float dyPixels);
    void zoom(float steps);

    void onMouseButton(int button, int action, int mods);
    void onCursorPos(double x, double y);
    void onScroll(double dx, double dy);
    void onResize(int width, int height);

    Mode  m_mode      = Mode::None;
    bool  m_firstMove = true;
    float m_lastX     = 0.0f;
    float m_lastY     = 0.0f;
};