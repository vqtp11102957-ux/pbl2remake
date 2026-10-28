#ifndef STRONGCLASSIFIER_H
#define STRONGCLASSIFIER_H

// ============================================================
// StrongClassifier.h - Bộ phân loại mạnh (một "tầng" của cascade)
//
// Gồm nhiều weak classifier, mỗi cái có trọng số alpha (do AdaBoost tính).
// Mỗi weak classifier bỏ phiếu vote = 0 hoặc 1 cho một cửa sổ:
//
//     score = sum( alpha_t * vote_t )
//     FACE (đi qua)  nếu  score >= threshold
//
// Ngưỡng mặc định theo bài báo:  threshold = 0.5 * sum(alpha_t)
// ("tổng phiếu có trọng số đạt ít nhất một nửa tổng trọng số tối đa").
//
// Trong cascade, mỗi tầng cần GIỮ HẦU HẾT các khuôn mặt (loại nhầm một mặt là
// mất luôn, không tầng sau nào cứu được), chấp nhận để lọt nhiều cửa sổ nền.
// Vì vậy ngưỡng được chọn theo MỤC TIÊU tỉ lệ giữ mặt (calibrateThreshold):
// tùy dữ liệu, ngưỡng đó có thể THẤP hơn hoặc CAO hơn ngưỡng mặc định.
// ============================================================

#include <iostream>
#include <vector>
#include "IntegralImage.h"
#include "TrainingSet.h"
#include "WeakClassifier.h"

// Giải thích chi tiết một lần chấm điểm (dùng cho Demo Mode).
struct StageTrace {
    std::vector<WeakTrace> weaks;   // phiếu của từng weak classifier
    double score;                   // sum(alpha * vote)
    double sumAlpha;                // điểm tối đa có thể đạt được
    double threshold;               // ngưỡng đang dùng
    bool passed;                    // score >= threshold
};

// Thống kê kết quả trên một tập mẫu.
struct RateStats {
    int faceCount;
    int nonFaceCount;
    int faceAccepted;               // mẫu FACE được cho đi qua (đúng)
    int nonFaceAccepted;            // mẫu NON_FACE bị cho đi qua (nhầm)
    double detectionRate;           // faceAccepted / faceCount
    double falsePositiveRate;       // nonFaceAccepted / nonFaceCount
    double errorRate;               // tỉ lệ mẫu bị đoán sai (cả hai loại)
};

class StrongClassifier {
private:
    std::vector<WeakClassifier> weaks;
    double threshold;

public:
    StrongClassifier();                                         // rỗng
    explicit StrongClassifier(const std::vector<WeakClassifier>& weakClassifiers);
    // (constructor thứ hai đặt ngưỡng = 0.5 * tổng alpha)

    // ---- Thông tin ----
    int size() const;                                // số weak classifier
    const WeakClassifier& getWeak(int index) const;  // index trong [0, size())
    double getSumAlpha() const;
    double getDefaultThreshold() const;              // 0.5 * getSumAlpha()
    double getThreshold() const;
    void setThreshold(double newThreshold);

    // ---- Chấm điểm một cửa sổ ----
    // (offsetX, offsetY): góc cửa sổ trong ảnh của ii; sigma: độ lệch chuẩn của cửa sổ.
    double score(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;
    bool predict(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;
    StageTrace explain(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;

    // ---- Đánh giá / chỉnh ngưỡng trên tập mẫu ----
    RateStats computeRates(const TrainingSet& data) const;

    // Chọn ngưỡng CAO NHẤT mà vẫn cho đi qua ít nhất targetDetectionRate các mẫu FACE
    // (ví dụ 0.99 = giữ lại 99% mặt). Trả về false nếu tập không có mẫu FACE.
    bool calibrateThreshold(const TrainingSet& data, double targetDetectionRate);

    // ---- Lưu / đọc ----
    //   THRESHOLD <số>
    //   WEAKS <n>
    //   <n dòng, mỗi dòng một weak classifier>
    void save(std::ostream& out) const;
    bool load(std::istream& in);     // false nếu dữ liệu sai; khi đó đối tượng không đổi
};

#endif
