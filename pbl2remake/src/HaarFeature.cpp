#include "HaarFeature.h"

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

HaarFeature::HaarFeature()
    : type(TWO_HORIZONTAL), x(0), y(0), w(1), h(1) {
}

HaarFeature::HaarFeature(HaarType type_, int x_, int y_, int w_, int h_)
    : type(type_), x(x_), y(y_), w(w_), h(h_) {
}

HaarFeature HaarFeature::fromRegion(HaarType type, const Rect& region) {
    HaarFeature probe(type, 0, 0, 1, 1);          // chỉ để hỏi số ô theo hai chiều
    int cellW = region.w / probe.cellsAcross();
    int cellH = region.h / probe.cellsDown();
    if (cellW < 1) cellW = 1;
    if (cellH < 1) cellH = 1;
    return HaarFeature(type, region.x, region.y, cellW, cellH);
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

HaarType HaarFeature::getType() const { return type; }
int HaarFeature::getX() const { return x; }
int HaarFeature::getY() const { return y; }
int HaarFeature::getW() const { return w; }
int HaarFeature::getH() const { return h; }

int HaarFeature::cellsAcross() const {
    switch (type) {
        case TWO_HORIZONTAL:   return 2;
        case TWO_VERTICAL:     return 1;
        case THREE_HORIZONTAL: return 3;
        case THREE_VERTICAL:   return 1;
        case FOUR_RECT:        return 2;
    }
    return 1;
}

int HaarFeature::cellsDown() const {
    switch (type) {
        case TWO_HORIZONTAL:   return 1;
        case TWO_VERTICAL:     return 2;
        case THREE_HORIZONTAL: return 1;
        case THREE_VERTICAL:   return 3;
        case FOUR_RECT:        return 2;
    }
    return 1;
}

int HaarFeature::getTotalWidth() const { return cellsAcross() * w; }
int HaarFeature::getTotalHeight() const { return cellsDown() * h; }

int HaarFeature::getBlackWeight() const {
    bool isThreeRect = (type == THREE_HORIZONTAL || type == THREE_VERTICAL);
    return isThreeRect ? 2 : 1;
}

bool HaarFeature::fitsInWindow(int windowSize) const {
    return w > 0 && h > 0 && x >= 0 && y >= 0 &&
           x + getTotalWidth() <= windowSize &&
           y + getTotalHeight() <= windowSize;
}

// ------------------------------------------------------------
// Hàm hỗ trợ private
// ------------------------------------------------------------

bool HaarFeature::isWhiteCell(int i, int j) {
    return (i + j) % 2 == 0;
}

Rect HaarFeature::cellRect(int i, int j, int offsetX, int offsetY) const {
    return Rect(offsetX + x + i * w, offsetY + y + j * h, w, h);
}

// ------------------------------------------------------------
// Tính giá trị feature
//
//   giá trị = (tổng các ô trắng) - hệ số * (tổng các ô đen)
//
// Mỗi ô được tính bằng IntegralImage::sumRect (4 phép tra bảng).
// ------------------------------------------------------------
long long HaarFeature::compute(const IntegralImage& ii, int offsetX, int offsetY) const {
    long long whiteSum = 0;
    long long blackSum = 0;

    for (int j = 0; j < cellsDown(); j++) {
        for (int i = 0; i < cellsAcross(); i++) {
            long long cellSum = ii.sumRect(cellRect(i, j, offsetX, offsetY));
            if (isWhiteCell(i, j)) {
                whiteSum += cellSum;
            } else {
                blackSum += cellSum;
            }
        }
    }
    return whiteSum - getBlackWeight() * blackSum;
}

// ------------------------------------------------------------
// Danh sách các hình chữ nhật trắng / đen (để vẽ và để demo)
// ------------------------------------------------------------

std::vector<Rect> HaarFeature::getWhiteRects(int offsetX, int offsetY) const {
    std::vector<Rect> result;
    for (int j = 0; j < cellsDown(); j++) {
        for (int i = 0; i < cellsAcross(); i++) {
            if (isWhiteCell(i, j)) {
                result.push_back(cellRect(i, j, offsetX, offsetY));
            }
        }
    }
    return result;
}

std::vector<Rect> HaarFeature::getBlackRects(int offsetX, int offsetY) const {
    std::vector<Rect> result;
    for (int j = 0; j < cellsDown(); j++) {
        for (int i = 0; i < cellsAcross(); i++) {
            if (!isWhiteCell(i, j)) {
                result.push_back(cellRect(i, j, offsetX, offsetY));
            }
        }
    }
    return result;
}

std::string HaarFeature::getTypeName() const {
    switch (type) {
        case TWO_HORIZONTAL:   return "TWO_HORIZONTAL";
        case TWO_VERTICAL:     return "TWO_VERTICAL";
        case THREE_HORIZONTAL: return "THREE_HORIZONTAL";
        case THREE_VERTICAL:   return "THREE_VERTICAL";
        case FOUR_RECT:        return "FOUR_RECT";
    }
    return "UNKNOWN";
}

std::string HaarFeature::describe() const {
    return getTypeName() + " x=" + std::to_string(x) + " y=" + std::to_string(y) +
           " cell=" + std::to_string(w) + "x" + std::to_string(h) +
           " total=" + std::to_string(getTotalWidth()) + "x" + std::to_string(getTotalHeight());
}
