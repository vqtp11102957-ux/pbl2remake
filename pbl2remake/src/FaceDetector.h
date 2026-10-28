#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

// ============================================================
// FaceDetector.h - Tìm khuôn mặt trong một ảnh bất kỳ
//
// Pipeline:
//   ảnh ──► (thu nhỏ nếu quá lớn) ──► grayscale
//        ──► KIM TỰ THÁP ẢNH: nhiều bản thu nhỏ liên tiếp (chia scaleFactor)
//              mỗi bản: tạo MỘT IntegralImage, rồi
//              CỬA SỔ TRƯỢT 24x24 quét toàn bản, mỗi lần dời `step` pixel,
//              mỗi cửa sổ đi qua CascadeClassifier
//        ──► các khung "thô" (một khuôn mặt thường được nhiều cửa sổ chồng nhau phát hiện)
//        ──► GỘP khung chồng nhau, chỉ giữ nhóm có >= minNeighbors khung
//        ──► danh sách hình chữ nhật (tọa độ trên ảnh GỐC)
//
// Vì sao dùng kim tự tháp? Cửa sổ luôn là 24x24, nên khuôn mặt to hơn 24 pixel phải
// được thu nhỏ lại cho vừa cửa sổ. Bản thu nhỏ k lần tìm được mặt rộng khoảng 24*k pixel.
// ============================================================

#include <string>
#include <vector>
#include "CascadeClassifier.h"
#include "Image.h"
#include "Types.h"

struct DetectorParams {
    double scaleFactor;        // tỉ lệ giữa hai mức kim tự tháp liên tiếp (> 1), mặc định 1.25
    int step;                  // bước trượt của cửa sổ (pixel, trên mỗi mức), mặc định 2
    int minNeighbors;          // một nhóm cần ít nhất ngần này khung thô mới được giữ, mặc định 2
                               // (cascade ngắn, chặt cho ít khung chồng nhau quanh mỗi mặt nên 3 dễ bỏ sót;
                               //  tăng lên nếu thấy nhiều khung báo nhầm, giảm xuống 1 nếu bỏ sót nhiều)
    double mergeOverlap;       // hai khung có IoU > giá trị này thì cùng nhóm, mặc định 0.3
    int minFaceSize;           // mặt nhỏ hơn (pixel trên ảnh gốc) bị bỏ qua, mặc định 24
    int maxFaceSize;           // mặt lớn hơn bị bỏ qua; 0 = không giới hạn
    int maxProcessingWidth;    // ảnh rộng hơn sẽ được thu nhỏ trước khi xử lý (0 = không thu nhỏ), mặc định 640

    DetectorParams()
        : scaleFactor(1.25), step(2), minNeighbors(2), mergeOverlap(0.3),
          minFaceSize(24), maxFaceSize(0), maxProcessingWidth(640) {}
};

struct DetectionStats {
    int levels;                // số mức kim tự tháp đã quét
    long long windowsEvaluated;
    int rawDetections;         // số khung thô trước khi gộp
    int mergedDetections;      // số khuôn mặt sau khi gộp
    double milliseconds;       // thời gian xử lý
};

class FaceDetector {
private:
    CascadeClassifier cascade;
    DetectorParams params;

    // Quét một mức kim tự tháp: trượt cửa sổ trên ii, ghi các khung thô vào `raw`.
    // (sx, sy): tỉ lệ để đổi tọa độ mức này về ảnh gốc.
    void scanLevel(const IntegralImage& ii, double sx, double sy,
                   std::vector<Rect>& raw, long long& windowCount) const;

    // Kim tự tháp ảnh: thu nhỏ dần, mỗi mức gọi scanLevel.
    void runPyramid(const Image& gray, double factor, std::vector<Rect>& raw,
                    long long& windowCount, int& levelCount) const;

    // Ảnh bất kỳ -> (thu nhỏ nếu quá lớn) -> xám -> kim tự tháp -> khung thô.
    std::vector<Rect> collectRaw(const Image& image, long long& windowCount, int& levelCount) const;

public:
    FaceDetector();
    explicit FaceDetector(const CascadeClassifier& cascade);

    // ---- Model ----
    void setCascade(const CascadeClassifier& newCascade);
    const CascadeClassifier& getCascade() const;
    bool hasModel() const;                         // có ít nhất một tầng
    bool loadModel(const std::string& path);       // false nếu không đọc được / sai định dạng
    bool saveModel(const std::string& path) const;

    // ---- Tham số ----
    const DetectorParams& getParams() const;
    void setParams(const DetectorParams& newParams);

    // ---- Phát hiện ----
    // Trả về các khuôn mặt (tọa độ trên ảnh gốc). Không có model hoặc ảnh rỗng -> danh sách rỗng.
    std::vector<Rect> detect(const Image& image, DetectionStats* stats = 0) const;

    // Như detect nhưng KHÔNG gộp: trả về mọi khung thô (dùng để minh họa).
    std::vector<Rect> detectRaw(const Image& image) const;

    // Gộp khung chồng nhau: khung có IoU > overlap thuộc cùng nhóm; nhóm có >= minNeighbors
    // khung thì được thay bằng khung trung bình; khung nhỏ nằm trong khung lớn hơn bị bỏ.
    static std::vector<Rect> mergeDetections(const std::vector<Rect>& raw,
                                             double overlap, int minNeighbors);
};

#endif
