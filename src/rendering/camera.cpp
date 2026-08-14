#include "camera.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

Camera::Camera(const glm::vec3& center, float dist)
    : target_(center), distance_(dist)
{
}

glm::vec3 Camera::position() const
{
    const float y = glm::radians(yaw_);
    const float p = glm::radians(pitch_);
    return target_ + distance_ * glm::vec3(std::cos(p) * std::cos(y),
                                         std::sin(p),
                                         std::cos(p) * std::sin(y));
}

glm::mat4 Camera::view() const
{
    return glm::lookAt(position(), target_, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::projection() const
{
    return glm::perspective(glm::radians(fov_), aspect_, znear_, zfar_);
}

glm::mat4 Camera::invProjection() const
{
    return glm::inverse(projection());
}

glm::mat4 Camera::invView() const
{
    return glm::inverse(view());
}

void Camera::rotate(float dxPixels, float dyPixels)
{
    yaw_   += dxPixels * rotateSpeed_;
    pitch_ += dyPixels * rotateSpeed_;
    pitch_  = std::clamp(pitch_, 0.0f, maxPitch_);
}

void Camera::zoom(float steps)
{
    distance_ *= std::pow(1.0f - zoomSpeed_, steps);
    distance_  = std::clamp(distance_, minDistance_, maxDistance_);
}

void Camera::onMouseButton(int button, int action, int mods)
{
    const bool down = (action == GLFW_PRESS);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mode_ = down ? Mode::Rotate : Mode::None;
    }

    firstMove_ = true;
}

void Camera::onCursorPos(double x, double y)
{
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);

    if (firstMove_)
    {
        lastX_ = fx;
        lastY_ = fy;
        firstMove_ = false;
        return;
    }

    const float dx = fx - lastX_;
    const float dy = fy - lastY_;
    lastX_ = fx;
    lastY_ = fy;

    if (mode_ == Mode::Rotate) rotate(dx, dy);
}

void Camera::onScroll(double /*dx*/, double dy)
{
    zoom(static_cast<float>(dy));
}

void Camera::onResize(int width, int height)
{
    if (height > 0)
        aspect_ = static_cast<float>(width) / static_cast<float>(height);
}

void Camera::attach(GLFWwindow* window)
{
    glfwSetWindowUserPointer(window, this);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int b, int a, int m) {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->onMouseButton(b, a, m);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->onCursorPos(x, y);
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double dx, double dy) {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->onScroll(dx, dy);
    });

    glfwSetWindowSizeCallback(window, [](GLFWwindow* w, int width, int high){
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->onResize(width, high);
    });

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    onResize(w, h);
}