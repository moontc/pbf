#pragma once
#include <cmath>

struct Vec3 {
    float x, y, z;
    Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(float s)      const { return Vec3(x*s, y*s, z*s); }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
};
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float len(const Vec3& a) { return std::sqrt(dot(a,a)); }

