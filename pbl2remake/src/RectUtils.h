#ifndef RECTUTILS_H
#define RECTUTILS_H

// RectUtils.h - Các phép tính nhỏ trên hình chữ nhật

#include "Types.h"

// Diện tích phần giao của hai hình chữ nhật (0 nếu không chồng nhau).
int intersectionArea(const Rect& a, const Rect& b);

// Độ chồng lấp IoU = diện tích giao / diện tích hợp, trong [0, 1].
double overlapRatio(const Rect& a, const Rect& b);

#endif
