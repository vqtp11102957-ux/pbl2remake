#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

// ============================================================
// IntegralImage.h - Bảng tích lũy (Integral Image)
//
// Với ảnh xám rộng W, cao H ta tạo bảng S có (W+1) x (H+1) ô:
//
//   S(x, y) = tổng mọi pixel nằm trong vùng [0, x) x [0, y)
//             (các cột 0..x-1 và các hàng 0..y-1)
//
//   => hàng 0 và cột 0 của S luôn bằng 0 (vùng rỗng).
//
// Tổng pixel của hình chữ nhật (x, y, w, h):
//
//        A ────── B          A = S(x    , y    )
//        │ vùng   │          B = S(x + w, y    )
//        C ────── D          C = S(x    , y + h)
//                            D = S(x + w, y + h)
//
//   tổng = D - B - C + A
//
// Ngoài bảng tổng pixel, ta còn lưu bảng tổng BÌNH PHƯƠNG pixel
// để tính độ lệch chuẩn của một cửa sổ (chuẩn hóa độ sáng, dùng ở
// các giai đoạn sau).
// ============================================================

#include <vector>
#include "Image.h"
#include "Types.h"

class IntegralImage {
private:
    int width;                          // chiều rộng ảnh gốc
    int height;                         // chiều cao ảnh gốc
    int stride;                         // số ô trên 1 hàng của bảng = width + 1
    std::vector<long long> sum;         // bảng tổng pixel,      (W+1)*(H+1) ô
    std::vector<long long> squareSum;   // bảng tổng pixel^2,    (W+1)*(H+1) ô

    void buildFromGray(const Image& gray);
    bool isValidRect(const Rect& r) const;
    long long rectSum(const std::vector<long long>& table, const Rect& r) const;

public:
    IntegralImage();                            // bảng rỗng
    explicit IntegralImage(const Image& image); // tạo luôn bảng từ ảnh

    // Tạo lại bảng. Nếu ảnh là ảnh màu thì tự đổi sang xám trước.
    void build(const Image& image);

    bool isEmpty() const;
    int  getWidth() const;
    int  getHeight() const;

    // Giá trị ô S(x, y) của bảng; x trong [0, width], y trong [0, height].
    long long getValue(int x, int y) const;

    // Tổng pixel trong hình chữ nhật. Hình vượt ra ngoài ảnh -> trả về 0.
    long long sumRect(const Rect& r) const;
    long long sumSquareRect(const Rect& r) const;

    // Trung bình và độ lệch chuẩn của các pixel trong hình chữ nhật.
    double windowMean(const Rect& r) const;
    double windowStdDev(const Rect& r) const;

    // Ảnh xám 8-bit để NHÌN bảng tích lũy (chia cho giá trị lớn nhất * 255).
    Image toVisualImage() const;
};

#endif
