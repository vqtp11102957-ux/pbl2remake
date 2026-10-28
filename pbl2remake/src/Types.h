#ifndef TYPES_H
#define TYPES_H

// ============================================================
// Types.h - Các kiểu dữ liệu nhỏ dùng chung cho cả project
// ============================================================

// Hình chữ nhật: góc trên-trái (x, y), rộng w, cao h.
// Quy ước của project: x = cột, y = hàng (gốc tọa độ ở góc trên-trái ảnh).
// LƯU Ý: không viết "using namespace cv;" vì OpenCV cũng có cv::Rect.
struct Rect {
    int x;
    int y;
    int w;
    int h;

    Rect() : x(0), y(0), w(0), h(0) {}
    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), w(w_), h(h_) {}
};

// Nhãn của một mẫu ảnh khi huấn luyện / kết quả dự đoán.
enum Label {
    NON_FACE = -1,
    FACE = 1
};

#endif
