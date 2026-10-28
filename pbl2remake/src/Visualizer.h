#ifndef VISUALIZER_H
#define VISUALIZER_H

// ============================================================
// Visualizer.h - Các hàm vẽ dùng cho giao diện và Demo Mode
//
// Tất cả đều TỰ VIẾT bằng thao tác trên pixel của class Image
// (không dùng hàm vẽ của OpenCV), nên có thể kiểm tra mà không cần cửa sổ.
// ============================================================

#include <vector>
#include "HaarFeature.h"
#include "Image.h"
#include "Types.h"

// Cách một ảnh được đặt vừa khung hiển thị: phóng/thu theo `scale`, đặt lệch (offsetX, offsetY).
//   điểm trên khung  = điểm trên ảnh * scale + offset
struct FitInfo {
    double scale;
    int offsetX;
    int offsetY;
    int drawWidth;       // kích thước ảnh sau khi phóng/thu
    int drawHeight;

    FitInfo() : scale(1.0), offsetX(0), offsetY(0), drawWidth(0), drawHeight(0) {}
};

class Visualizer {
public:
    // Ảnh màu 3 kênh (ảnh xám được nhân ra 3 kênh).
    static Image toColor(const Image& img);

    // Vẽ viền chữ nhật dày `thickness` pixel (phần ngoài ảnh bị cắt bỏ).
    static void drawOutline(Image& img, const Rect& r, int red, int green, int blue, int thickness);

    // Tô nửa trong suốt: màu mới = (1 - percent%) * màu cũ + percent% * (red, green, blue).
    static void tint(Image& img, const Rect& r, int red, int green, int blue, int percent);

    // Đặt ảnh vừa khung areaW x areaH (giữ tỉ lệ, căn giữa, nền xám tối).
    // Kết quả luôn là ảnh màu kích thước đúng areaW x areaH.
    static Image fitInto(const Image& img, int areaW, int areaH, FitInfo* info = 0);

    // Đổi hình chữ nhật từ tọa độ ảnh sang tọa độ khung (theo fit), và ngược lại cho một điểm.
    static Rect mapRect(const Rect& r, const FitInfo& fit);
    static void viewToImage(const FitInfo& fit, int viewX, int viewY, int& imageX, int& imageY);

    // Ảnh với các khung bao mặt (đặt vừa khung areaW x areaH).
    static Image renderBoxes(const Image& img, const std::vector<Rect>& boxes,
                             int areaW, int areaH, int red, int green, int blue);

    // Ảnh xám (hoặc màu) đặt vừa khung, kèm vùng TRẮNG của feature tô xanh lá, vùng ĐEN tô đỏ.
    // Tọa độ feature là tọa độ trên chính ảnh `img`.
    static Image renderFeature(const Image& img, const HaarFeature& feature,
                               int areaW, int areaH, FitInfo* info = 0);
};

#endif
