#pragma once
#include <cmath>

struct Vec3 {
    double x, y, z;
    Vec3(double a = 0, double b = 0, double c = 0) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(double s)      const { return Vec3(x*s, y*s, z*s); }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
};
inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline double len(const Vec3& a) { return std::sqrt(dot(a,a)); }

