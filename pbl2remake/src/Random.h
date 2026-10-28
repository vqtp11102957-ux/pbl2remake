#ifndef RANDOM_H
#define RANDOM_H

// ============================================================
// Random.h - Bộ sinh số ngẫu nhiên đơn giản (LCG)
//
// Tự viết thay vì dùng rand() để kết quả GIỐNG HỆT NHAU trên mọi máy và mọi
// trình biên dịch (huấn luyện lại với cùng seed cho cùng một model).
// ============================================================

class Random {
private:
    unsigned int state;

public:
    explicit Random(unsigned int seed = 12345) : state(seed) {}

    // Số nguyên trong [0, n). n phải trong khoảng 1..65536.
    int nextInt(int n) {
        state = state * 1664525u + 1013904223u;
        return static_cast<int>((state >> 16) % static_cast<unsigned int>(n));
    }
};

#endif
