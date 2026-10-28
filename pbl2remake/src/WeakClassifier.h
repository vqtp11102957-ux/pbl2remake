#ifndef WEAKCLASSIFIER_H
#define WEAKCLASSIFIER_H

// ============================================================
// WeakClassifier.h - Bộ phân loại yếu ("decision stump")
//
// Một weak classifier chỉ dùng MỘT Haar feature và MỘT ngưỡng:
//
//   1. Tính giá trị feature trên cửa sổ, rồi CHUẨN HÓA ĐỘ SÁNG:
//          value = giá_trị_feature / sigma        (sigma = độ lệch chuẩn cửa sổ)
//   2. Bỏ phiếu:
//          vote = 1 (mặt)      nếu  polarity * value  <  polarity * threshold
//          vote = 0 (không)    ngược lại
//
//   polarity = +1 : mặt khi value NHỎ HƠN ngưỡng
//   polarity = -1 : mặt khi value LỚN HƠN ngưỡng
//
// "alpha" là trọng số (độ tin cậy) do AdaBoost gán ở giai đoạn sau;
// phiếu có trọng số của weak classifier này là  alpha * vote.
// ============================================================

#include <iostream>
#include <vector>
#include "HaarFeature.h"
#include "IntegralImage.h"
#include "Types.h"

// Giải thích chi tiết một lần bỏ phiếu (dùng cho Demo Mode).
struct WeakTrace {
    HaarFeature feature;
    long long rawValue;      // giá trị feature thô (tổng trắng - hệ số * tổng đen)
    double sigma;            // độ lệch chuẩn của cửa sổ
    double featureValue;     // giá trị đã chuẩn hóa = rawValue / sigma
    double threshold;
    int polarity;
    int vote;                // 1 = mặt, 0 = không phải mặt
    double alpha;
};

// Kết quả tìm ngưỡng tốt nhất cho một feature.
struct ThresholdResult {
    double threshold;
    int polarity;            // +1 hoặc -1
    double error;            // tổng trọng số các mẫu bị đoán sai
};

class WeakClassifier {
private:
    HaarFeature feature;
    double threshold;
    int polarity;            // +1 hoặc -1
    double alpha;

public:
    WeakClassifier();
    WeakClassifier(const HaarFeature& feature, double threshold,
                   int polarity, double alpha = 0.0);

    // ---- Thông tin ----
    const HaarFeature& getFeature() const;
    double getThreshold() const;
    int getPolarity() const;
    double getAlpha() const;
    void setAlpha(double newAlpha);

    // ---- Dự đoán ----
    // Chuẩn hóa giá trị thô theo sigma (sigma < 1 được coi là 1 để tránh chia cho 0).
    static double normalize(long long rawValue, double sigma);

    // Giá trị feature đã chuẩn hóa trên cửa sổ có góc (offsetX, offsetY).
    double featureValue(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;

    // Phiếu (0 hoặc 1) khi đã biết giá trị feature đã chuẩn hóa.
    int voteOnValue(double normalizedValue) const;

    // Phiếu (0 hoặc 1) cho cửa sổ.
    int predict(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;

    // Như predict nhưng trả về đầy đủ các bước trung gian.
    WeakTrace explain(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const;

    // ---- Tìm ngưỡng (dùng khi huấn luyện) ----
    // Trả về thứ tự chỉ số sao cho values[order[0]] <= values[order[1]] <= ...
    static std::vector<int> sortedOrder(const std::vector<double>& values);

    // Tìm (ngưỡng, polarity) có tổng trọng số lỗi nhỏ nhất.
    //   values[i]  : giá trị feature (đã chuẩn hóa) của mẫu i
    //   order      : kết quả của sortedOrder(values)
    //   labels[i]  : FACE hoặc NON_FACE
    //   weights[i] : trọng số của mẫu i
    static ThresholdResult findBestThreshold(const std::vector<double>& values,
                                             const std::vector<int>& order,
                                             const std::vector<Label>& labels,
                                             const std::vector<double>& weights);

    // ---- Lưu / đọc (một dòng văn bản) ----
    void save(std::ostream& out) const;
    bool load(std::istream& in);       // false nếu dữ liệu sai; khi đó đối tượng không đổi
};

#endif
