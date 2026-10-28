#include "RectUtils.h"

#include <algorithm>

int intersectionArea(const Rect& a, const Rect& b) {
    int left = std::max(a.x, b.x);
    int top = std::max(a.y, b.y);
    int right = std::min(a.x + a.w, b.x + b.w);
    int bottom = std::min(a.y + a.h, b.y + b.h);
    if (right <= left || bottom <= top) {
        return 0;
    }
    return (right - left) * (bottom - top);
}

double overlapRatio(const Rect& a, const Rect& b) {
    int inter = intersectionArea(a, b);
    int unionArea = a.w * a.h + b.w * b.h - inter;
    if (unionArea <= 0) {
        return 0.0;
    }
    return static_cast<double>(inter) / unionArea;
}
