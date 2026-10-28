#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

// ============================================================
// CascadeClassifier.h - Cascade: nhiều tầng StrongClassifier nối tiếp
//
//   cửa sổ ──► [phẳng?] ──► tầng 1 ──► tầng 2 ──► tầng 3 ──► ... ──► MẶT
//                │            │          │          │
//              loại         loại       loại       loại        (loại ở đâu thì DỪNG ở đó)
//
// Vì sao nhanh? Hầu hết cửa sổ trong một ảnh là nền (tường, trời, quần áo...)
// và bị loại ngay ở tầng 1 hoặc 2, chỉ cần chạy vài weak classifier. Chỉ những
// cửa sổ "đáng ngờ" mới đi sâu vào các tầng sau (nhiều weak classifier hơn).
//
// Một cửa sổ là MẶT khi và chỉ khi nó vượt qua TẤT CẢ các tầng.
//
// Trước các tầng còn có một phép loại nhanh: cửa sổ gần như phẳng
// (sigma < minSigma) không thể là khuôn mặt (không có sáng/tối để so sánh).
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include "IntegralImage.h"
#include "StrongClassifier.h"
#include "Types.h"

// Giải thích chi tiết đường đi của một cửa sổ qua cascade (dùng cho Demo Mode).
struct WindowTrace {
    Rect box;                        // vị trí cửa sổ trong ảnh
    double sigma;                    // độ lệch chuẩn của cửa sổ
    bool flatRejected;               // true nếu bị loại vì quá phẳng (sigma < minSigma)
    std::vector<StageTrace> stages;  // các tầng ĐÃ chạy (dừng ngay tầng làm loại cửa sổ)
    int rejectedAtStage;             // chỉ số tầng (từ 0) làm loại cửa sổ; -1 nếu không bị tầng nào loại
    bool accepted;                   // true nếu là MẶT

    WindowTrace()
        : box(), sigma(0.0), flatRejected(false), stages(), rejectedAtStage(-1), accepted(false) {}
};

class CascadeClassifier {
private:
    int windowSize;                  // cửa sổ vuông windowSize x windowSize
    double minSigma;                 // cửa sổ có sigma nhỏ hơn giá trị này bị loại ngay
    std::vector<StrongClassifier> stages;

public:
    explicit CascadeClassifier(int windowSize = 24, double minSigma = 4.0);

    // ---- Xây dựng ----
    void addStage(const StrongClassifier& stage);
    void clear();

    // ---- Thông tin ----
    int getWindowSize() const;
    double getMinSigma() const;
    void setMinSigma(double newMinSigma);
    int stageCount() const;
    const StrongClassifier& getStage(int index) const;   // index trong [0, stageCount())
    int totalWeakCount() const;
    std::string summary() const;                         // ví dụ: "3 tang [2, 5, 10], tong 17 weak classifier"

    // ---- Phân loại một cửa sổ ----
    // Cửa sổ có góc trên-trái (offsetX, offsetY) và kích thước windowSize x windowSize
    // trong ảnh của ii. Cửa sổ nằm ngoài ảnh -> false.
    // Nếu trace khác nullptr thì ghi lại đường đi chi tiết (chậm hơn một chút).
    bool classifyWindow(const IntegralImage& ii, int offsetX, int offsetY,
                        WindowTrace* trace = 0) const;

    // ---- Lưu / đọc ----
    //   VJ_MODEL 1
    //   WINDOW <n>
    //   MINSIGMA <số>
    //   STAGES <số tầng>
    //   STAGE 0
    //   <nội dung StrongClassifier>
    //   STAGE 1 ...
    void save(std::ostream& out) const;
    bool load(std::istream& in);     // false nếu sai; khi đó đối tượng không đổi
};

#endif
