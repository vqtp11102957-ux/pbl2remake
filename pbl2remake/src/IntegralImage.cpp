#include "IntegralImage.h"

#include <cmath>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

IntegralImage::IntegralImage()
    : width(0), height(0), stride(1), sum(), squareSum() {
}

IntegralImage::IntegralImage(const Image& image)
    : width(0), height(0), stride(1), sum(), squareSum() {
    build(image);
}

// ------------------------------------------------------------
// Tạo bảng tích lũy
// ------------------------------------------------------------

void IntegralImage::build(const Image& image) {
    if (image.isEmpty()) {
        width = 0;
        height = 0;
        stride = 1;
        sum.clear();
        squareSum.clear();
        return;
    }

    if (image.getChannels() == 1) {
        buildFromGray(image);
    } else {
        buildFromGray(image.toGrayscale());
    }
}

// Công thức xây bảng (đi từ trên xuống, từ trái sang phải):
//
//   S(x+1, y+1) = (tổng pixel hàng y, từ cột 0 đến cột x)   <- "rowSum"
//               + S(x+1, y)                                  <- tổng của các hàng phía trên
//
// Hàng 0 và cột 0 của bảng giữ nguyên giá trị 0.
void IntegralImage::buildFromGray(const Image& gray) {
    width = gray.getWidth();
    height = gray.getHeight();
    stride = width + 1;

    size_t tableSize = static_cast<size_t>(stride) * (height + 1);
    sum.assign(tableSize, 0);
    squareSum.assign(tableSize, 0);

    for (int y = 0; y < height; y++) {
        long long rowSum = 0;          // cộng dồn pixel của hàng y
        long long rowSquareSum = 0;    // cộng dồn pixel^2 của hàng y

        for (int x = 0; x < width; x++) {
            long long p = gray.getPixel(x, y);
            rowSum += p;
            rowSquareSum += p * p;

            int here  = (y + 1) * stride + (x + 1);   // ô S(x+1, y+1)
            int above = y * stride + (x + 1);         // ô S(x+1, y)
            sum[here] = rowSum + sum[above];
            squareSum[here] = rowSquareSum + squareSum[above];
        }
    }
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

bool IntegralImage::isEmpty() const { return sum.empty(); }
int IntegralImage::getWidth() const { return width; }
int IntegralImage::getHeight() const { return height; }

long long IntegralImage::getValue(int x, int y) const {
    if (x < 0 || x > width || y < 0 || y > height || isEmpty()) {
        return 0;
    }
    return sum[y * stride + x];
}

// ------------------------------------------------------------
// Tổng vùng bằng 4 phép tra bảng:  D - B - C + A
// ------------------------------------------------------------

bool IntegralImage::isValidRect(const Rect& r) const {
    return !isEmpty() && r.w > 0 && r.h > 0 &&
           r.x >= 0 && r.y >= 0 &&
           r.x + r.w <= width && r.y + r.h <= height;
}

long long IntegralImage::rectSum(const std::vector<long long>& table, const Rect& r) const {
    if (!isValidRect(r)) {
        return 0;
    }
    long long A = table[r.y * stride + r.x];                 // góc trên-trái
    long long B = table[r.y * stride + (r.x + r.w)];         // góc trên-phải
    long long C = table[(r.y + r.h) * stride + r.x];         // góc dưới-trái
    long long D = table[(r.y + r.h) * stride + (r.x + r.w)]; // góc dưới-phải
    return D - B - C + A;
}

long long IntegralImage::sumRect(const Rect& r) const {
    return rectSum(sum, r);
}

long long IntegralImage::sumSquareRect(const Rect& r) const {
    return rectSum(squareSum, r);
}

// ------------------------------------------------------------
// Trung bình và độ lệch chuẩn của một cửa sổ
//
//   mean     = (tổng pixel) / (số pixel)
//   variance = (tổng pixel^2) / (số pixel) - mean^2
//   stddev   = sqrt(variance)
// ------------------------------------------------------------

double IntegralImage::windowMean(const Rect& r) const {
    if (!isValidRect(r)) {
        return 0.0;
    }
    double area = static_cast<double>(r.w) * r.h;
    return sumRect(r) / area;
}

double IntegralImage::windowStdDev(const Rect& r) const {
    if (!isValidRect(r)) {
        return 0.0;
    }
    double area = static_cast<double>(r.w) * r.h;
    double mean = sumRect(r) / area;
    double variance = sumSquareRect(r) / area - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;              // tránh sai số làm tròn ra số âm
    }
    return std::sqrt(variance);
}

// ------------------------------------------------------------
// Ảnh để quan sát bảng tích lũy
// Ô S(x+1, y+1) tương ứng pixel (x, y) của ảnh gốc.
// Ô lớn nhất là góc dưới-phải = tổng toàn bộ ảnh -> map thành 255.
// ------------------------------------------------------------

Image IntegralImage::toVisualImage() const {
    if (isEmpty()) {
        return Image();
    }

    Image visual(width, height, 1);
    long long maxValue = sum[height * stride + width];
    if (maxValue == 0) {
        return visual;               // ảnh toàn đen
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            long long v = sum[(y + 1) * stride + (x + 1)];
            visual.setPixel(x, y, 0, static_cast<unsigned char>(v * 255 / maxValue));
        }
    }
    return visual;
}
