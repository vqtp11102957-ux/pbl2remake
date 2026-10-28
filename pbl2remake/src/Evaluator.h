#ifndef EVALUATOR_H
#define EVALUATOR_H

// ============================================================
// Evaluator.h - Đo chất lượng bộ nhận diện trên tập ảnh KIỂM TRA
//
// Cấu trúc thư mục:
//   dataset/test/positive/   ảnh CÓ khuôn mặt
//   dataset/test/negative/   ảnh KHÔNG có khuôn mặt
//
// Đánh giá ở mức ẢNH (mỗi ảnh một kết quả):
//   - ảnh positive mà detector tìm được ít nhất 1 mặt    -> TP (phát hiện đúng)
//   - ảnh positive mà detector không tìm được mặt nào    -> FN (bỏ sót)
//   - ảnh negative mà detector tìm được ít nhất 1 "mặt"  -> FP (báo nhầm)
//   - ảnh negative mà detector không tìm thấy gì         -> TN (đúng)
//
//   Accuracy  = (TP + TN) / tổng số ảnh
//   Precision = TP / (TP + FP)      (trong các ảnh báo "có mặt", bao nhiêu ảnh thật sự có mặt)
//   Recall    = TP / (TP + FN)      (trong các ảnh thật sự có mặt, bao nhiêu ảnh được tìm ra)
// ============================================================

#include <string>
#include <vector>
#include "FaceDetector.h"
#include "Image.h"
#include "ProgressListener.h"

struct EvaluationResult {
    int positiveImages;
    int negativeImages;
    int truePositives;
    int falseNegatives;
    int falsePositives;
    int trueNegatives;
    int facesFound;               // tổng số khung mặt tìm được trên mọi ảnh
    double accuracy;              // 0..1
    double precision;             // 0..1 (0 nếu TP + FP = 0)
    double recall;                // 0..1 (0 nếu TP + FN = 0)
    double totalMilliseconds;
    double averageMilliseconds;   // trung bình mỗi ảnh
    std::vector<std::string> mistakes;   // tên các ảnh bị sai (tối đa 20)

    EvaluationResult()
        : positiveImages(0), negativeImages(0), truePositives(0), falseNegatives(0),
          falsePositives(0), trueNegatives(0), facesFound(0), accuracy(0.0), precision(0.0),
          recall(0.0), totalMilliseconds(0.0), averageMilliseconds(0.0), mistakes() {}

    int totalImages() const { return positiveImages + negativeImages; }
};

class Evaluator {
private:
    // Chạy detector trên một ảnh, cập nhật kết quả. name chỉ dùng để ghi lại ảnh bị sai.
    static void processImage(const FaceDetector& detector, const Image& image, bool hasFace,
                             const std::string& name, EvaluationResult& result);

    // Tính Accuracy / Precision / Recall / thời gian trung bình từ các số đếm.
    static void finish(EvaluationResult& result);

public:
    // Đánh giá trên ảnh đã đọc sẵn.
    static EvaluationResult evaluateImages(const FaceDetector& detector,
                                           const std::vector<Image>& positives,
                                           const std::vector<Image>& negatives,
                                           ProgressListener* listener = 0);

    // Đọc testFolder/positive và testFolder/negative rồi đánh giá. Trả về kết quả rỗng
    // (totalImages() == 0) nếu thư mục không có ảnh nào.
    static EvaluationResult evaluateFolders(const FaceDetector& detector,
                                            const std::string& testFolder,
                                            ProgressListener* listener = 0);

    // Báo cáo văn bản nhiều dòng (chỉ ký tự ASCII).
    static std::vector<std::string> describe(const EvaluationResult& result);

    // Ghi báo cáo ra file. false nếu không ghi được.
    static bool saveReport(const EvaluationResult& result, const std::string& path);
};

#endif
