#ifndef DEMOCONTROLLER_H
#define DEMOCONTROLLER_H

// ============================================================
// DemoController.h - Chế độ trình diễn thuật toán từng bước
//
// 9 bước, bấm "Next Step" để đi tiếp:
//   1. Original image          6. Weak classifier (phiếu của weak classifier đầu tiên)
//   2. Grayscale               7. Strong classifier (tầng 1: cộng phiếu có trọng số)
//   3. Integral Image          8. Cascade (cửa sổ đi qua / bị loại ở tầng nào)
//   4. Chọn một Haar feature   9. Kết quả phát hiện (khung bao mặt)
//   5. Tính giá trị feature
//
// Để minh họa, ta chọn MỘT cửa sổ: khuôn mặt lớn nhất mà detector tìm được
// (nếu không có mặt nào thì lấy cửa sổ ở giữa ảnh), thu về 24x24, rồi cho đi qua
// ĐÚNG các hàm thật của chương trình (không có bản "diễn" riêng).
// ============================================================

#include <string>
#include <vector>
#include "FaceDetector.h"
#include "HaarFeature.h"
#include "Image.h"
#include "IntegralImage.h"
#include "WeakClassifier.h"

class DemoController {
public:
    static constexpr int STEP_COUNT = 9;

private:
    bool prepared;
    int step;                       // 0 .. STEP_COUNT - 1

    Image original;
    Image gray;
    Image integralVisual;           // Integral Image của cả ảnh, chuẩn hóa về 0..255 để nhìn
    Rect windowBox;                 // cửa sổ được chọn, trên ảnh gốc
    bool faceFound;                 // cửa sổ này là khuôn mặt do detector tìm ra?
    Image window;                   // cửa sổ thu về windowSize x windowSize (xám)
    IntegralImage windowIntegral;
    double windowSigma;

    HaarFeature feature;            // weak classifier đầu tiên của tầng 1
    WeakTrace weakTrace;
    StageTrace firstStageTrace;
    WindowTrace cascadeTrace;
    int windowSize;
    double minSigma;
    std::vector<Rect> finalBoxes;   // kết quả phát hiện trên toàn ảnh

public:
    DemoController();

    // Chạy toàn bộ pipeline trên ảnh. false nếu ảnh rỗng hoặc detector chưa có model.
    bool prepare(const Image& image, const FaceDetector& detector);

    bool isPrepared() const;
    void reset();                   // về bước 1
    bool next();                    // false nếu đang ở bước cuối
    bool previous();                // false nếu đang ở bước đầu
    int getStep() const;            // 0..8

    std::string getTitle() const;                 // ví dụ "Step 3/9: Integral Image"
    std::vector<std::string> getLines() const;   // lời giải thích của bước hiện tại (ASCII)

    // Hình minh họa của bước hiện tại, đúng kích thước areaW x areaH.
    Image render(int areaW, int areaH) const;
};

#endif
