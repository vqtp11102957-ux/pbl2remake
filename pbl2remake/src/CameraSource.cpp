#include "CameraSource.h"

CameraSource::CameraSource() : capture() {
}

CameraSource::~CameraSource() {
    close();
}

bool CameraSource::open(int index) {
    close();
#ifdef _WIN32
    // Trên Windows, DirectShow thường mở webcam nhanh hơn backend mặc định.
    if (capture.open(index, cv::CAP_DSHOW) && capture.isOpened()) {
        return true;
    }
    capture.release();
#endif
    return capture.open(index) && capture.isOpened();
}

void CameraSource::close() {
    if (capture.isOpened()) {
        capture.release();
    }
}

bool CameraSource::isOpened() const {
    return capture.isOpened();
}

bool CameraSource::read(Image& frame) {
    if (!capture.isOpened()) {
        return false;
    }
    cv::Mat mat;
    if (!capture.read(mat) || mat.empty()) {
        return false;
    }
    Image converted = Image::fromMat(mat);
    if (converted.isEmpty()) {
        return false;
    }
    frame = converted;
    return true;
}
