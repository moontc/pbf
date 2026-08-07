#pragma once
#include <cmath>

#ifdef __CUDACC__
#define PBF_HD __host__ __device__
#else
#define PBF_HD

#endif

struct Vec3 {
    float x, y, z;
    PBF_HD Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
    PBF_HD Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    PBF_HD Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    PBF_HD Vec3 operator*(float s)      const { return Vec3(x*s, y*s, z*s); }
    PBF_HD Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
};
PBF_HD inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

PBF_HD inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y*b.z - a.z*b.y,
                a.z*b.x - a.x*b.z,
                a.x*b.y - a.y*b.x);
}

PBF_HD inline float len(const Vec3& a) { return sqrtf(dot(a,a)); }