#pragma once
template <typename T>
class Point3 {
public:
    // Point3 Public Methods
    Point3() { x = y = z = 0; }
    Point3(T x, T y, T z) : x(x), y(y), z(z) { assert(!HasNaNs()); }
    template <typename U>
    explicit Point3(const Point3<U>& p)
        : x((T)p.x), y((T)p.y), z((T)p.z) {
        assert(!HasNaNs());
    }
    template <typename U>
    explicit operator Vector3<U>() const {
        return Vector3<U>(x, y, z);
    }
#ifndef NDEBUG
    Point3(const Point3<T>& p) {
        assert(!p.HasNaNs());
        x = p.x;
        y = p.y;
        z = p.z;
    }

    Point3<T>& operator=(const Point3<T>& p) {
        assert(!p.HasNaNs());
        x = p.x;
        y = p.y;
        z = p.z;
        return *this;
    }
#endif  // !NDEBUG
    Point3<T> operator+(const Vector3<T>& v) const {
        assert(!v.HasNaNs());
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }
    Point3<T>& operator+=(const Vector3<T>& v) {
        assert(!v.HasNaNs());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vector3<T> operator-(const Point3<T>& p) const {
        assert(!p.HasNaNs());
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }
    Point3<T> operator-(const Vector3<T>& v) const {
        assert(!v.HasNaNs());
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }
    Point3<T>& operator-=(const Vector3<T>& v) {
        assert(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Point3<T>& operator+=(const Point3<T>& p) {
        assert(!p.HasNaNs());
        x += p.x;
        y += p.y;
        z += p.z;
        return *this;
    }
    Point3<T> operator+(const Point3<T>& p) const {
        assert(!p.HasNaNs());
        return Point3<T>(x + p.x, y + p.y, z + p.z);
    }
    template <typename U>
    Point3<T> operator*(U f) const {
        return Point3<T>(f * x, f * y, f * z);
    }
    template <typename U>
    Point3<T>& operator*=(U f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }
    template <typename U>
    Point3<T> operator/(U f) const {
        CHECK_NE(f, 0);
        Float inv = (Float)1 / f;
        return Point3<T>(inv * x, inv * y, inv * z);
    }
    template <typename U>
    Point3<T>& operator/=(U f) {
        CHECK_NE(f, 0);
        Float inv = (Float)1 / f;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }
    T operator[](int i) const {
        assert(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    T& operator[](int i) {
        assert(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    bool operator==(const Point3<T>& p) const {
        return x == p.x && y == p.y && z == p.z;
    }
    bool operator!=(const Point3<T>& p) const {
        return x != p.x || y != p.y || z != p.z;
    }
    bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
    Point3<T> operator-() const { return Point3<T>(-x, -y, -z); }

    // Point3 Public Data
    T x, y, z;
};

typedef Point3<float> Point3f;
typedef Point3<int> Point3i;