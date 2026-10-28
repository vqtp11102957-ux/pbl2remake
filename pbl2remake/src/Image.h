#ifndef IMAGE_H
#define IMAGE_H

// ============================================================
// Image.h - Class lưu ảnh và các phép biến đổi cơ bản
//
// Quy ước lưu trữ:
//   - Ảnh xám  : channels = 1, mỗi pixel 1 byte (0..255)
//   - Ảnh màu  : channels = 3, mỗi pixel 3 byte theo thứ tự R, G, B
//   - Pixel (x, y) kênh c nằm ở data[(y * width + x) * channels + c]
//   - x = cột (0..width-1), y = hàng (0..height-1)
//
// OpenCV chỉ được dùng cho: đọc/ghi file, resize, và chuyển qua lại
// cv::Mat để hiển thị. Việc chuyển ảnh màu sang xám do ta tự viết.
// ============================================================

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "Types.h"

class Image {
private:
    int width;
    int height;
    int channels;                     // 1 = xám, 3 = màu RGB
    std::vector<unsigned char> data;  // width * height * channels byte

    // Đổi (x, y, kênh) thành chỉ số trong mảng 1 chiều.
    int indexOf(int x, int y, int c) const;

public:
    // ---- Constructor ----
    Image();                                              // ảnh rỗng
    Image(int w, int h, int ch, unsigned char fillValue = 0);

    // ---- Đọc / ghi file (dùng OpenCV) ----
    bool load(const std::string& path);   // true nếu đọc được
    bool save(const std::string& path) const;

    // ---- Thông tin ----
    int  getWidth() const;
    int  getHeight() const;
    int  getChannels() const;
    bool isEmpty() const;
    bool isInside(int x, int y) const;

    // ---- Truy cập pixel ----
    unsigned char getPixel(int x, int y, int c = 0) const;
    void setPixel(int x, int y, int c, unsigned char value);

    // ---- Biến đổi ảnh (trả về ảnh MỚI, không sửa ảnh hiện tại) ----
    Image toGrayscale() const;                    // TỰ CÀI ĐẶT
    Image crop(const Rect& r) const;              // TỰ CÀI ĐẶT
    Image resized(int newWidth, int newHeight) const;  // dùng cv::resize
    Image flippedHorizontally() const;            // lật trái <-> phải

    // ---- Vẽ lên ảnh (dùng cho giao diện) ----
    // Tô hình chữ nhật (phần ngoài ảnh bị cắt bỏ). Ảnh xám chỉ dùng giá trị red.
    void fillRect(const Rect& r, unsigned char red, unsigned char green, unsigned char blue);
    // Dán ảnh src lên ảnh này, góc trên-trái của src đặt tại (x, y). Phần ra ngoài bị cắt.
    // Tự đổi giữa ảnh xám và ảnh màu khi số kênh khác nhau.
    void paste(const Image& src, int x, int y);

    // ---- Cầu nối với OpenCV (webcam, hiển thị) ----
    // mat phải là ảnh 8-bit, 1 kênh (xám) hoặc 3 kênh (BGR như OpenCV).
    static Image fromMat(const cv::Mat& mat);
    cv::Mat toMat() const;
};

#endif
