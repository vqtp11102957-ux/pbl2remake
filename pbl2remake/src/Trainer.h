#ifndef TRAINER_H
#define TRAINER_H

// ============================================================
// Trainer.h - Huấn luyện CascadeClassifier từ dataset
//
//   dataset/positive  ──► ảnh mặt ──► (lật ngang để nhân đôi) ──► xáo trộn
//                                      ├─ phần HUẤN LUYỆN  (học AdaBoost)
//                                      └─ phần KIỂM ĐỊNH   (chỉnh ngưỡng từng tầng)
//   dataset/negative  ──► ảnh nền ──► cắt thật nhiều cửa sổ 24x24 ──► "kho" nền
//
//   Lặp từng tầng:
//     1. Lấy các cửa sổ nền trong kho mà các tầng TRƯỚC còn cho lọt qua (bootstrapping)
//     2. AdaBoost học tầng mới từ (mặt huấn luyện) + (nền còn lọt)
//     3. Chỉnh ngưỡng tầng để giữ targetDetectionRate mặt KIỂM ĐỊNH
//     4. Thêm tầng vào cascade
//
// Đây là phiên bản rút gọn mang tính giáo dục (xem README): ít tầng, tập dữ liệu nhỏ,
// số weak classifier mỗi tầng cố định.
// ============================================================

#include <string>
#include <vector>
#include "CascadeClassifier.h"
#include "Image.h"
#include "ProgressListener.h"

struct TrainConfig {
    int windowSize;                       // kích thước cửa sổ (24)
    int positionStep;                     // bước vị trí khi sinh Haar feature
    int sizeStep;                         // bước kích thước
    int minCellSize;                      // ô nhỏ nhất
    std::vector<int> stageWeakCounts;     // số weak classifier của từng tầng, ví dụ {2, 5, 10, 20, 30}
    double targetDetectionRate;           // tỉ lệ mặt kiểm định mỗi tầng phải giữ lại
    double validationFraction;            // phần mặt dành cho kiểm định
    int maxPositives;                     // số ảnh mặt tối đa đọc từ thư mục (<= 0: hết)
    bool addMirroredPositives;            // thêm bản lật ngang của mỗi mặt
    int maxTrainFaces;                    // số mặt huấn luyện tối đa mỗi tầng
    int maxNegativeImages;                // số ảnh nền tối đa đọc từ thư mục
    int negativeWindowsPerImage;          // số cửa sổ cắt từ mỗi ảnh nền
    int maxNegativePool;                  // kích thước tối đa của "kho" cửa sổ nền
    int negativesPerStage;                // số cửa sổ nền dùng để học mỗi tầng
    int minNegativesToContinue;           // còn ít hơn số này thì dừng thêm tầng
    bool mineHardNegatives;               // quét lại ảnh nền bằng cascade hiện tại để tìm thêm cửa sổ bị báo nhầm
    int miningStep;                       // bước trượt khi quét tìm (pixel)
    double minSigma;                      // cửa sổ phẳng hơn mức này bị loại ngay
    unsigned int seed;                    // hạt giống ngẫu nhiên

    TrainConfig()
        : windowSize(24), positionStep(2), sizeStep(2), minCellSize(4),
          stageWeakCounts(), targetDetectionRate(0.995), validationFraction(0.25),
          maxPositives(1500), addMirroredPositives(true), maxTrainFaces(2000),
          maxNegativeImages(1000), negativeWindowsPerImage(30), maxNegativePool(8000),
          negativesPerStage(1500), minNegativesToContinue(100), mineHardNegatives(true), miningStep(4),
          minSigma(4.0), seed(12345) {
        stageWeakCounts.push_back(2);
        stageWeakCounts.push_back(5);
        stageWeakCounts.push_back(10);
        stageWeakCounts.push_back(20);
        stageWeakCounts.push_back(30);
    }
};

struct StageReport {
    int weakCount;
    double threshold;
    double validationDetection;           // tỉ lệ mặt kiểm định mà TẦNG NÀY giữ lại
    int poolBefore;                       // số cửa sổ nền còn lọt qua các tầng trước (đã gồm cả cửa sổ mới tìm thêm)
    int poolAfter;                        // ... sau khi thêm tầng này
    int mined;                            // số cửa sổ nền "khó" mới tìm thêm từ ảnh nền cho tầng này
};

struct TrainReport {
    bool success;
    bool cancelled;
    std::string message;                  // lời nhắn (lý do thất bại, ...)
    int positivesRead;                    // số ảnh mặt đã đọc
    int trainFaces;
    int validationFaces;
    int negativePool;                     // số cửa sổ nền trong kho (kể cả các cửa sổ tìm thêm)
    int minedTotal;                       // tổng số cửa sổ nền "khó" tìm thêm trong cả quá trình
    int featureCount;
    std::vector<StageReport> stages;
    double finalDetection;                // tỉ lệ mặt kiểm định mà CẢ cascade giữ lại
    double finalPoolFalsePositive;        // tỉ lệ cửa sổ nền trong kho còn lọt qua CẢ cascade
    double seconds;

    TrainReport()
        : success(false), cancelled(false), message(), positivesRead(0), trainFaces(0),
          validationFaces(0), negativePool(0), minedTotal(0), featureCount(0), stages(),
          finalDetection(0.0), finalPoolFalsePositive(0.0), seconds(0.0) {}
};

class Trainer {
private:
    TrainConfig config;

public:
    explicit Trainer(const TrainConfig& config = TrainConfig());

    const TrainConfig& getConfig() const;

    // Huấn luyện từ ảnh đã đọc sẵn. positives: ảnh mặt (kích thước bất kỳ);
    // negativeImages: ảnh nền (kích thước bất kỳ). listener có thể để trống.
    CascadeClassifier trainFromImages(const std::vector<Image>& positives,
                                      const std::vector<Image>& negativeImages,
                                      ProgressListener* listener, TrainReport& report) const;

    // Đọc datasetFolder/positive và datasetFolder/negative rồi huấn luyện.
    CascadeClassifier trainFromFolders(const std::string& datasetFolder,
                                       ProgressListener* listener, TrainReport& report) const;

    // Báo cáo dạng văn bản (nhiều dòng, chỉ dùng ký tự ASCII).
    static std::vector<std::string> describe(const TrainReport& report);
};

#endif
