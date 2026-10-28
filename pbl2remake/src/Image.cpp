#include "Image.h"

#include <algorithm>

#include <opencv2/imgcodecs.hpp>   // cv::imread, cv::imwrite
#include <opencv2/imgproc.hpp>     // cv::resize

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

// Ảnh rỗng: chưa có pixel nào.
Image::Image() : width(0), height(0), channels(0), data() {
}

// Tạo ảnh w x h, ch kênh, mọi byte đều bằng fillValue.
// Nếu tham số sai (kích thước <= 0, ch khác 1 và 3) thì tạo ảnh rỗng.
Image::Image(int w, int h, int ch, unsigned char fillValue)
    : width(0), height(0), channels(0), data() {
    bool valid = (w > 0 && h > 0 && (ch == 1 || ch == 3));
    if (valid) {
        width = w;
        height = h;
        channels = ch;
        data.assign(static_cast<size_t>(w) * h * ch, fillValue);
    }
}

// ------------------------------------------------------------
// Hàm hỗ trợ private
// ------------------------------------------------------------

int Image::indexOf(int x, int y, int c) const {
    return (y * width + x) * channels + c;
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

int Image::getWidth() const { return width; }
int Image::getHeight() const { return height; }
int Image::getChannels() const { return channels; }
bool Image::isEmpty() const { return data.empty(); }

bool Image::isInside(int x, int y) const {
    return x >= 0 && x < width && y >= 0 && y < height;
}

// ------------------------------------------------------------
// Truy cập pixel
// ------------------------------------------------------------

// Ngoài ảnh hoặc kênh sai thì trả về 0 (không làm chương trình sập).
unsigned char Image::getPixel(int x, int y, int c) const {
    if (!isInside(x, y) || c < 0 || c >= channels) {
        return 0;
    }
    return data[indexOf(x, y, c)];
}

void Image::setPixel(int x, int y, int c, unsigned char value) {
    if (!isInside(x, y) || c < 0 || c >= channels) {
        return;
    }
    data[indexOf(x, y, c)] = value;
}

// ------------------------------------------------------------
// Chuyển ảnh màu sang ảnh xám (TỰ CÀI ĐẶT)
//
//   gray = 0.299 * R + 0.587 * G + 0.114 * B
//
// Để tránh số thực, ta nhân cả công thức với 1000 rồi chia lại,
// cộng thêm 500 để làm tròn thay vì cắt bỏ phần thập phân.
// ------------------------------------------------------------
Image Image::toGrayscale() const {
    if (isEmpty()) {
        return Image();
    }
    if (channels == 1) {
        return *this;               // đã là ảnh xám: trả về bản sao
    }

    Image gray(width, height, 1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = data[indexOf(x, y, 0)];
            int g = data[indexOf(x, y, 1)];
            int b = data[indexOf(x, y, 2)];
            int value = (299 * r + 587 * g + 114 * b + 500) / 1000;
            gray.data[y * width + x] = static_cast<unsigned char>(value);
        }
    }
    return gray;
}

// ------------------------------------------------------------
// Cắt một vùng chữ nhật. Vùng phải nằm TRỌN trong ảnh,
// nếu không trả về ảnh rỗng.
// ------------------------------------------------------------
Image Image::crop(const Rect& r) const {
    bool valid = !isEmpty() && r.w > 0 && r.h > 0 &&
                 r.x >= 0 && r.y >= 0 &&
                 r.x + r.w <= width && r.y + r.h <= height;
    if (!valid) {
        return Image();
    }

    Image result(r.w, r.h, channels);
    for (int y = 0; y < r.h; y++) {
        for (int x = 0; x < r.w; x++) {
            for (int c = 0; c < channels; c++) {
                result.data[result.indexOf(x, y, c)] = data[indexOf(r.x + x, r.y + y, c)];
            }
        }
    }
    return result;
}

// ------------------------------------------------------------
// Đổi kích thước (dùng cv::resize - được phép theo đề bài).
// Thu nhỏ thì dùng INTER_AREA cho mịn, phóng to dùng INTER_LINEAR.
// ------------------------------------------------------------
Image Image::resized(int newWidth, int newHeight) const {
    if (isEmpty() || newWidth <= 0 || newHeight <= 0) {
        return Image();
    }

    bool shrinking = (newWidth < width) || (newHeight < height);
    int method = shrinking ? cv::INTER_AREA : cv::INTER_LINEAR;

    cv::Mat src = toMat();
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(newWidth, newHeight), 0, 0, method);
    return fromMat(dst);
}

// ------------------------------------------------------------
// Lật ngang
// ------------------------------------------------------------
Image Image::flippedHorizontally() const {
    if (isEmpty()) {
        return Image();
    }
    Image result(width, height, channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                result.data[result.indexOf(width - 1 - x, y, c)] = data[indexOf(x, y, c)];
            }
        }
    }
    return result;
}

// ------------------------------------------------------------
// Tô hình chữ nhật
// ------------------------------------------------------------
void Image::fillRect(const Rect& r, unsigned char red, unsigned char green, unsigned char blue) {
    if (isEmpty()) {
        return;
    }
    int x0 = std::max(0, r.x);
    int y0 = std::max(0, r.y);
    int x1 = std::min(width, r.x + r.w);
    int y1 = std::min(height, r.y + r.h);
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int index = indexOf(x, y, 0);
            data[index] = red;
            if (channels == 3) {
                data[index + 1] = green;
                data[index + 2] = blue;
            }
        }
    }
}

// ------------------------------------------------------------
// Dán một ảnh lên ảnh này
// ------------------------------------------------------------
void Image::paste(const Image& src, int x, int y) {
    if (isEmpty() || src.isEmpty()) {
        return;
    }
    for (int sy = 0; sy < src.height; sy++) {
        int dy = y + sy;
        if (dy < 0 || dy >= height) {
            continue;
        }
        for (int sx = 0; sx < src.width; sx++) {
            int dx = x + sx;
            if (dx < 0 || dx >= width) {
                continue;
            }
            int from = src.indexOf(sx, sy, 0);
            int to = indexOf(dx, dy, 0);
            if (channels == src.channels) {
                for (int c = 0; c < channels; c++) {
                    data[to + c] = src.data[from + c];
                }
            } else if (channels == 3) {                       // src xám -> ảnh màu
                data[to] = data[to + 1] = data[to + 2] = src.data[from];
            } else {                                          // src màu -> ảnh xám
                int r = src.data[from], g = src.data[from + 1], b = src.data[from + 2];
                data[to] = static_cast<unsigned char>((299 * r + 587 * g + 114 * b + 500) / 1000);
            }
        }
    }
}

// ------------------------------------------------------------
// Đọc / ghi file
// ------------------------------------------------------------

// cv::imread với IMREAD_COLOR luôn trả về ảnh 3 kênh BGR,
// kể cả khi file gốc là ảnh xám hoặc có kênh alpha.
bool Image::load(const std::string& path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);
    if (mat.empty()) {
        return false;
    }
    *this = fromMat(mat);
    return !isEmpty();
}

bool Image::save(const std::string& path) const {
    if (isEmpty()) {
        return false;
    }
    return cv::imwrite(path, toMat());
}

// ------------------------------------------------------------
// Chuyển qua lại với cv::Mat
//
// cv::Mat là kiểu ảnh của OpenCV:
//   mat.rows = số hàng  = height
//   mat.cols = số cột   = width
//   mat.ptr<unsigned char>(y) = con trỏ tới đầu hàng y
//   ảnh 3 kênh lưu mỗi pixel là B, G, R (ngược với Image của ta!)
// Ta copy từng pixel, đồng thời đảo B<->R cho đúng quy ước.
// ------------------------------------------------------------
Image Image::fromMat(const cv::Mat& mat) {
    if (mat.empty() || mat.depth() != CV_8U) {
        return Image();
    }
    int ch = mat.channels();
    if (ch != 1 && ch != 3) {
        return Image();
    }

    Image img(mat.cols, mat.rows, ch);
    for (int y = 0; y < mat.rows; y++) {
        const unsigned char* row = mat.ptr<unsigned char>(y);
        for (int x = 0; x < mat.cols; x++) {
            if (ch == 1) {
                img.data[img.indexOf(x, y, 0)] = row[x];
            } else {
                img.data[img.indexOf(x, y, 0)] = row[3 * x + 2];  // R
                img.data[img.indexOf(x, y, 1)] = row[3 * x + 1];  // G
                img.data[img.indexOf(x, y, 2)] = row[3 * x + 0];  // B
            }
        }
    }
    return img;
}

cv::Mat Image::toMat() const {
    if (isEmpty()) {
        return cv::Mat();
    }

    cv::Mat mat(height, width, channels == 1 ? CV_8UC1 : CV_8UC3);
    for (int y = 0; y < height; y++) {
        unsigned char* row = mat.ptr<unsigned char>(y);
        for (int x = 0; x < width; x++) {
            if (channels == 1) {
                row[x] = data[indexOf(x, y, 0)];
            } else {
                row[3 * x + 0] = data[indexOf(x, y, 2)];  // B
                row[3 * x + 1] = data[indexOf(x, y, 1)];  // G
                row[3 * x + 2] = data[indexOf(x, y, 0)];  // R
            }
        }
    }
    return mat;
}
