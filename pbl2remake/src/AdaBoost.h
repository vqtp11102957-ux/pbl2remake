#ifndef ADABOOST_H
#define ADABOOST_H

// ============================================================
// AdaBoost.h - Thuật toán AdaBoost (phiên bản của bài báo Viola-Jones)
//
// Input : tập mẫu (FACE / NON_FACE) + danh sách Haar feature
// Output: T weak classifier, mỗi cái có trọng số alpha
//
// Ý tưởng: mỗi mẫu có một TRỌNG SỐ. Mỗi vòng:
//   1. Chuẩn hóa trọng số (tổng = 1).
//   2. Với MỖI feature, tìm ngưỡng tốt nhất -> lỗi có trọng số.
//   3. Chọn feature có lỗi epsilon nhỏ nhất -> weak classifier của vòng này.
//   4. alpha = ln((1 - epsilon) / epsilon)       (lỗi càng nhỏ, alpha càng lớn)
//   5. Mẫu được đoán ĐÚNG nhân trọng số với beta = epsilon / (1 - epsilon) < 1
//      (mẫu bị đoán sai giữ nguyên) -> sau khi chuẩn hóa, mẫu khó có trọng số lớn hơn.
//   6. Lặp lại.
//
// Strong classifier: FACE nếu  sum(alpha_t * vote_t)  >=  0.5 * sum(alpha_t)
// (class StrongClassifier đầy đủ được viết ở giai đoạn sau)
// ============================================================

#include <vector>
#include "HaarFeature.h"
#include "ProgressListener.h"
#include "TrainingSet.h"
#include "WeakClassifier.h"

// Ghi lại những gì xảy ra ở mỗi vòng (để báo cáo và demo).
struct RoundInfo {
    int round;                  // 1, 2, 3, ...
    int featureIndex;           // vị trí feature được chọn trong danh sách đưa vào
    double error;               // epsilon: lỗi có trọng số của weak classifier được chọn
    double alpha;               // ln((1 - epsilon) / epsilon)
    double errorAfterUpdate;    // lỗi của chính nó với trọng số MỚI (lý thuyết: đúng 0.5)
    double ensembleError;       // tỉ lệ mẫu bị strong classifier (các vòng đến nay) đoán sai
    double detectionRate;       // tỉ lệ mẫu FACE được nhận đúng
    double falsePositiveRate;   // tỉ lệ mẫu NON_FACE bị nhận nhầm là mặt
};

struct AdaBoostResult {
    std::vector<WeakClassifier> weakClassifiers;   // đã có alpha
    std::vector<RoundInfo> history;
    bool cancelled;                                // true nếu người dùng hủy giữa chừng

    AdaBoostResult() : weakClassifiers(), history(), cancelled(false) {}
};

class AdaBoost {
private:
    int rounds;     // số vòng lặp tối đa = số weak classifier tối đa

    // Bước chuẩn bị: giá trị (đã chuẩn hóa) của MỌI feature trên MỌI mẫu,
    // và thứ tự sắp xếp các mẫu theo giá trị của từng feature (làm MỘT lần).
    // Trả về false nếu người dùng hủy (qua listener).
    static bool computeFeatureValues(const TrainingSet& data,
                                     const std::vector<HaarFeature>& features,
                                     std::vector<std::vector<double> >& values,
                                     ProgressListener* listener);
    static void computeSortedOrders(const std::vector<std::vector<double> >& values,
                                    std::vector<std::vector<int> >& orders);

    // Bước 2-3 của mỗi vòng: tìm feature (cùng ngưỡng, polarity) có lỗi nhỏ nhất.
    static int findBestFeature(const std::vector<std::vector<double> >& values,
                               const std::vector<std::vector<int> >& orders,
                               const std::vector<Label>& labels,
                               const std::vector<double>& weights,
                               ThresholdResult& bestResult);

public:
    explicit AdaBoost(int rounds = 10);
    int getRounds() const;

    // Trọng số khởi tạo: mỗi mẫu FACE được 1/(2*số FACE), mỗi mẫu NON_FACE được
    // 1/(2*số NON_FACE) -> hai lớp có tổng trọng số bằng nhau (mỗi lớp 0.5).
    static std::vector<double> initialWeights(const std::vector<Label>& labels);

    // Chia mọi trọng số cho tổng để tổng bằng 1.
    static void normalizeWeights(std::vector<double>& weights);

    // Chạy AdaBoost. Trả về kết quả rỗng nếu không thể huấn luyện
    // (không có feature, rounds < 1, hoặc thiếu mẫu FACE / NON_FACE).
    // Dừng sớm nếu weak classifier tốt nhất cũng chỉ ngang đoán ngẫu nhiên.
    // listener (có thể bỏ trống): nhận tiến trình 0..100% và cho phép hủy.
    AdaBoostResult train(const TrainingSet& data, const std::vector<HaarFeature>& features,
                         ProgressListener* listener = 0) const;
};

#endif
