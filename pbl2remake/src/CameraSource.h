#ifndef CAMERASOURCE_H
#define CAMERASOURCE_H

// ============================================================
// CameraSource.h - Webcam (bọc cv::VideoCapture)
//
// OpenCV chỉ được dùng để LẤY KHUNG HÌNH. Khung hình được đổi ngay sang Image;
// từ đó trở đi mọi xử lý (grayscale, Integral Image, cascade...) là code của ta.
// ============================================================

#include <opencv2/videoio.hpp>
#include "Image.h"

class CameraSource {
private:
    cv::VideoCapture capture;

    // Không cho sao chép (không thể chia sẻ một webcam giữa hai đối tượng).
    CameraSource(const CameraSource&);
    CameraSource& operator=(const CameraSource&);

public:
    CameraSource();
    ~CameraSource();

    // Mở webcam số `index` (0 = webcam mặc định). Trả về false nếu không mở được.
    bool open(int index = 0);
    void close();
    bool isOpened() const;

    // Đọc một khung hình. false nếu webcam chưa mở hoặc không đọc được.
    bool read(Image& frame);
};

#endif
