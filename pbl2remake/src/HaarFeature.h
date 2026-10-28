#ifndef HAARFEATURE_H
#define HAARFEATURE_H

// ============================================================
// HaarFeature.h - Một đặc trưng Haar-like (Haar feature)
//
// Một feature là một lưới các ô chữ nhật BẰNG NHAU, tô màu kiểu bàn cờ:
//   ô (i, j) có (i + j) chẵn  -> vùng TRẮNG
//   ô (i, j) có (i + j) lẻ    -> vùng ĐEN
//
//   TWO_HORIZONTAL   TWO_VERTICAL    THREE_HORIZONTAL    THREE_VERTICAL     FOUR_RECT
//   (2 ô ngang)      (2 ô dọc)       (3 ô ngang)         (3 ô dọc)          (4 ô, 2x2)
//   +---+---+        +---+           +---+---+---+       +---+              +---+---+
//   | W | B |        | W |           | W | B | W |       | W |              | W | B |
//   +---+---+        +---+           +---+---+---+       +---+              +---+---+
//                    | B |                               | B |              | B | W |
//                    +---+                               +---+              +---+---+
//                                                        | W |
//                                                        +---+
//
// Giá trị feature = (tổng pixel vùng trắng) - hệ số * (tổng pixel vùng đen)
//   hệ số = 2 cho feature 3 ô (ô giữa hẹp hơn tổng hai ô ngoài nên nhân đôi
//           để hai bên "cân diện tích": vùng ảnh phẳng cho giá trị 0)
//   hệ số = 1 cho các loại còn lại
//
// Vị trí (x, y) và kích thước (w, h) của MỘT ô được tính trong hệ tọa độ
// của CỬA SỔ (ví dụ cửa sổ 24x24), không phải của cả ảnh.
// ============================================================

#include <string>
#include <vector>
#include "IntegralImage.h"
#include "Types.h"

enum HaarType {
    TWO_HORIZONTAL   = 0,
    TWO_VERTICAL     = 1,
    THREE_HORIZONTAL = 2,
    THREE_VERTICAL   = 3,
    FOUR_RECT        = 4
};

class HaarFeature {
private:
    HaarType type;
    int x;      // cột của góc trên-trái feature, trong cửa sổ
    int y;      // hàng của góc trên-trái feature, trong cửa sổ
    int w;      // chiều rộng của MỘT ô
    int h;      // chiều cao của MỘT ô

    // Hình chữ nhật của ô (i, j), dời theo vị trí cửa sổ (offsetX, offsetY).
    Rect cellRect(int i, int j, int offsetX, int offsetY) const;

    // Ô (i, j) là vùng trắng khi (i + j) chẵn.
    static bool isWhiteCell(int i, int j);

public:
    HaarFeature();
    HaarFeature(HaarType type, int x, int y, int w, int h);

    // Tạo feature loại `type` vừa khít một vùng chữ nhật (chia đều vùng thành các ô;
    // phần dư bị bỏ). Dùng khi người dùng kéo chuột chọn vùng trên ảnh.
    static HaarFeature fromRegion(HaarType type, const Rect& region);

    // ---- Thông tin ----
    HaarType getType() const;
    int getX() const;
    int getY() const;
    int getW() const;
    int getH() const;

    int cellsAcross() const;      // số ô theo chiều ngang (1, 2 hoặc 3)
    int cellsDown() const;        // số ô theo chiều dọc   (1, 2 hoặc 3)
    int getTotalWidth() const;    // cellsAcross() * w
    int getTotalHeight() const;   // cellsDown() * h
    int getBlackWeight() const;   // 2 cho feature 3 ô, 1 cho loại còn lại

    // Feature có nằm trọn trong cửa sổ windowSize x windowSize không?
    bool fitsInWindow(int windowSize) const;

    // ---- Tính giá trị ----
    // ii là IntegralImage của ảnh; (offsetX, offsetY) là góc trên-trái của
    // cửa sổ trong ảnh đó (khi chỉ có một cửa sổ thì để 0, 0).
    long long compute(const IntegralImage& ii, int offsetX = 0, int offsetY = 0) const;

    // ---- Phục vụ hiển thị / demo ----
    std::vector<Rect> getWhiteRects(int offsetX = 0, int offsetY = 0) const;
    std::vector<Rect> getBlackRects(int offsetX = 0, int offsetY = 0) const;
    std::string getTypeName() const;
    std::string describe() const;
};

#endif
