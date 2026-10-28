#include "Trainer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>

#include "AdaBoost.h"
#include "DatasetLoader.h"
#include "FeatureGenerator.h"
#include "Random.h"
#include "StrongClassifier.h"
#include "TrainingSet.h"

namespace {

// Đổi tiến trình 0..100% của một việc con thành khoảng [low, high] của việc cha.
// (Ví dụ AdaBoost chạy 0..100% nhưng chỉ chiếm phần 30%..50% của cả quá trình huấn luyện.)
class ScaledProgress : public ProgressListener {
private:
    ProgressListener* outer;
    int low;
    int high;
    std::string prefix;

public:
    ScaledProgress(ProgressListener* outer_, int low_, int high_, const std::string& prefix_)
        : outer(outer_), low(low_), high(high_), prefix(prefix_) {}

    void onProgress(const std::string& message, int percent) {
        if (outer != 0) {
            outer->onProgress(prefix + message, low + (high - low) * percent / 100);
        }
    }

    bool shouldCancel() {
        return outer != 0 && outer->shouldCancel();
    }
};

void notify(ProgressListener* listener, const std::string& message, int percent) {
    if (listener != 0) {
        listener->onProgress(message, percent);
    }
}

// Các cửa sổ nền trong kho mà cascade hiện tại còn cho lọt qua.
std::vector<int> survivors(const CascadeClassifier& cascade, const TrainingSet& pool) {
    std::vector<int> alive;
    for (int i = 0; i < pool.size(); i++) {
        if (cascade.classifyWindow(pool.getImage(i), 0, 0)) {
            alive.push_back(i);
        }
    }
    return alive;
}

// HARD NEGATIVE MINING: quét các ảnh nền (kim tự tháp + cửa sổ trượt) bằng cascade HIỆN TẠI và
// lấy các cửa sổ mà cascade VẪN cho là mặt (tức là báo nhầm). Đó chính là những mẫu nền "khó" mà
// tầng kế tiếp cần học để loại bỏ. Mỗi ảnh chỉ cho một phần nhỏ để mẫu đa dạng.
std::vector<Image> mineHardNegatives(const CascadeClassifier& cascade, const std::vector<Image>& images,
                                     int needed, int step, unsigned int seed) {
    std::vector<Image> found;
    if (needed <= 0 || images.empty()) {
        return found;
    }
    int windowSize = cascade.getWindowSize();
    int count = static_cast<int>(images.size());
    Random random(seed);
    int start = random.nextInt(count < 65536 ? count : 65536);
    int perImage = std::max(3, 2 * needed / count);

    for (int k = 0; k < count && static_cast<int>(found.size()) < needed; k++) {
        const Image& image = images[(start + k) % count];
        int taken = 0;
        for (double scale = 1.0; taken < perImage && static_cast<int>(found.size()) < needed; scale *= 1.25) {
            int levelW = static_cast<int>(image.getWidth() / scale + 0.5);
            int levelH = static_cast<int>(image.getHeight() / scale + 0.5);
            if (levelW < windowSize || levelH < windowSize) {
                break;
            }
            Image level = (scale == 1.0) ? image : image.resized(levelW, levelH);
            IntegralImage ii(level);
            for (int y = 0; y + windowSize <= levelH && taken < perImage; y += step) {
                for (int x = 0; x + windowSize <= levelW && taken < perImage; x += step) {
                    if (cascade.classifyWindow(ii, x, y)) {
                        found.push_back(level.crop(Rect(x, y, windowSize, windowSize)));
                        taken++;
                        x += windowSize / 2;               // bỏ qua các cửa sổ sát ngay cạnh (gần như trùng)
                    }
                }
            }
        }
    }
    return found;
}

// Xáo trộn (Fisher-Yates) bằng bộ ngẫu nhiên có seed cố định.
void shuffleImages(std::vector<Image>& images, Random& random) {
    for (int i = static_cast<int>(images.size()) - 1; i > 0; i--) {
        int j = random.nextInt(i + 1);
        std::swap(images[i], images[j]);
    }
}

}  // namespace

Trainer::Trainer(const TrainConfig& config_) : config(config_) {
}

const TrainConfig& Trainer::getConfig() const { return config; }

CascadeClassifier Trainer::trainFromImages(const std::vector<Image>& positives,
                                           const std::vector<Image>& negativeImages,
                                           ProgressListener* listener, TrainReport& report) const {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    report = TrainReport();
    CascadeClassifier cascade(config.windowSize, config.minSigma);
    report.positivesRead = static_cast<int>(positives.size());

    if (positives.size() < 20) {
        report.message = "Need at least 20 positive (face) images.";
        return cascade;
    }
    if (negativeImages.empty()) {
        report.message = "No negative images found.";
        return cascade;
    }

    // ---- 1. Chuẩn bị mặt: về 24x24, lật ngang, xáo trộn, chia huấn luyện / kiểm định ----
    notify(listener, "Preparing face samples", 1);
    std::vector<Image> faces;
    for (size_t i = 0; i < positives.size(); i++) {
        Image face = DatasetLoader::prepareFace(positives[i], config.windowSize);
        if (face.isEmpty()) {
            continue;
        }
        faces.push_back(face);
        if (config.addMirroredPositives) {
            faces.push_back(face.flippedHorizontally());
        }
    }
    Random random(config.seed);
    shuffleImages(faces, random);

    int validationCount = static_cast<int>(faces.size() * config.validationFraction);
    if (validationCount < 10) {
        validationCount = 10;
    }
    int trainCount = static_cast<int>(faces.size()) - validationCount;
    if (trainCount < 10) {
        report.message = "Not enough face samples after splitting.";
        return cascade;
    }
    if (trainCount > config.maxTrainFaces) {
        trainCount = config.maxTrainFaces;
    }
    std::vector<Image> trainFaces(faces.begin(), faces.begin() + trainCount);
    TrainingSet validationFaces(config.windowSize);
    for (size_t i = faces.size() - validationCount; i < faces.size(); i++) {
        validationFaces.add(faces[i], FACE);
    }
    report.trainFaces = trainCount;
    report.validationFaces = validationCount;

    // ---- 2. Kho cửa sổ nền ----
    notify(listener, "Cutting negative windows", 3);
    std::vector<Image> negatives = DatasetLoader::sampleNegativeWindows(
        negativeImages, config.windowSize, config.negativeWindowsPerImage, config.maxNegativePool, config.seed + 1);
    if (negatives.size() < 50) {
        report.message = "Not enough negative windows (negative images are too small or too few).";
        return cascade;
    }
    TrainingSet negativePool(config.windowSize);
    for (size_t i = 0; i < negatives.size(); i++) {
        negativePool.add(negatives[i], NON_FACE);
    }
    std::vector<Image> grayNegativeImages;                 // ảnh nền (xám) dùng để tìm thêm cửa sổ khó
    for (size_t i = 0; i < negativeImages.size(); i++) {
        grayNegativeImages.push_back(negativeImages[i].toGrayscale());
    }

    // ---- 3. Haar feature ----
    std::vector<HaarFeature> features =
        FeatureGenerator(config.windowSize, config.positionStep, config.sizeStep, config.minCellSize).generate();
    report.featureCount = static_cast<int>(features.size());

    // ---- 4. Từng tầng ----
    int stageTotal = static_cast<int>(config.stageWeakCounts.size());
    for (int s = 0; s < stageTotal; s++) {
        int low = 5 + 93 * s / stageTotal;
        int high = 5 + 93 * (s + 1) / stageTotal;
        std::string prefix = "Stage " + std::to_string(s + 1) + "/" + std::to_string(stageTotal) + ": ";
        notify(listener, prefix + "selecting hard negatives", low);
        if (listener != 0 && listener->shouldCancel()) {
            report.cancelled = true;
            break;
        }

        std::vector<int> alive = survivors(cascade, negativePool);

        // Kho cạn dần: quét lại ảnh nền bằng cascade hiện tại để tìm thêm cửa sổ bị báo nhầm
        int mined = 0;
        if (config.mineHardNegatives && cascade.stageCount() > 0 &&
            static_cast<int>(alive.size()) < config.negativesPerStage) {
            notify(listener, prefix + "mining hard negatives from the images", low);
            std::vector<Image> extra = mineHardNegatives(cascade, grayNegativeImages,
                                                         config.negativesPerStage - static_cast<int>(alive.size()),
                                                         config.miningStep, config.seed + 100 + s);
            for (size_t i = 0; i < extra.size(); i++) {
                negatives.push_back(extra[i]);
                negativePool.add(extra[i], NON_FACE);
                alive.push_back(negativePool.size() - 1);
            }
            mined = static_cast<int>(extra.size());
            report.minedTotal += mined;
        }
        if (static_cast<int>(alive.size()) < config.minNegativesToContinue) {
            report.message = "Stopped early: only " + std::to_string(alive.size()) +
                             " negative windows still pass the cascade.";
            break;
        }

        TrainingSet data(config.windowSize);
        for (size_t i = 0; i < trainFaces.size(); i++) {
            data.add(trainFaces[i], FACE);
        }
        for (size_t i = 0; i < alive.size() && static_cast<int>(i) < config.negativesPerStage; i++) {
            data.add(negatives[alive[i]], NON_FACE);
        }

        ScaledProgress inner(listener, low, high, prefix);
        AdaBoostResult result = AdaBoost(config.stageWeakCounts[s]).train(data, features, &inner);
        if (result.cancelled) {
            report.cancelled = true;
            break;
        }
        if (result.weakClassifiers.empty()) {
            report.message = "AdaBoost could not find a useful weak classifier for stage " + std::to_string(s + 1) + ".";
            break;
        }

        StrongClassifier stage(result.weakClassifiers);
        stage.calibrateThreshold(validationFaces, config.targetDetectionRate);
        cascade.addStage(stage);

        StageReport stageReport;
        stageReport.weakCount = stage.size();
        stageReport.threshold = stage.getThreshold();
        stageReport.validationDetection = stage.computeRates(validationFaces).detectionRate;
        stageReport.poolBefore = static_cast<int>(alive.size());
        stageReport.poolAfter = static_cast<int>(survivors(cascade, negativePool).size());
        stageReport.mined = mined;
        report.stages.push_back(stageReport);
    }

    // ---- 5. Kết quả ----
    report.negativePool = negativePool.size();
    int accepted = 0;
    for (int i = 0; i < validationFaces.size(); i++) {
        if (cascade.classifyWindow(validationFaces.getImage(i), 0, 0)) {
            accepted++;
        }
    }
    report.finalDetection = static_cast<double>(accepted) / validationFaces.size();
    report.finalPoolFalsePositive = static_cast<double>(survivors(cascade, negativePool).size()) / negativePool.size();
    report.success = !report.cancelled && cascade.stageCount() > 0;
    if (report.success && report.message.empty()) {
        report.message = "Training finished.";
    }
    // Mỗi tầng chỉ được phép bỏ floor((1 - target) * số mặt kiểm định) mặt. Nếu con số đó bằng 0
    // thì ngưỡng tầng bằng điểm của mặt "lạ" nhất và tầng có thể mất tác dụng lọc.
    if (report.success && validationFaces.size() * (1.0 - config.targetDetectionRate) < 1.0) {
        report.message += " Warning: only " + std::to_string(validationFaces.size()) +
                          " validation faces for a " + std::to_string(static_cast<int>(config.targetDetectionRate * 1000) / 10.0).substr(0, 4) +
                          "% target: stage thresholds equal the lowest face score. Use more faces.";
    }
    if (cascade.stageCount() == 0 && report.message.empty()) {
        report.message = "No stage was trained.";
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    report.seconds = std::chrono::duration<double>(end - start).count();
    notify(listener, report.cancelled ? "Cancelled" : "Done", 100);
    return cascade;
}

CascadeClassifier Trainer::trainFromFolders(const std::string& datasetFolder,
                                            ProgressListener* listener, TrainReport& report) const {
    std::filesystem::path root(datasetFolder);
    std::string positiveFolder = (root / "positive").string();
    std::string negativeFolder = (root / "negative").string();

    notify(listener, "Reading positive images", 0);
    int skippedPositive = 0, skippedNegative = 0;
    std::vector<Image> positives = DatasetLoader::loadPositives(positiveFolder, config.windowSize,
                                                                config.maxPositives, &skippedPositive);
    notify(listener, "Reading negative images", 1);
    std::vector<Image> negatives = DatasetLoader::loadNegativeImages(negativeFolder, config.maxNegativeImages,
                                                                     &skippedNegative);

    if (positives.empty() || negatives.empty()) {
        report = TrainReport();
        report.message = "Dataset folder must contain 'positive' and 'negative' sub-folders with images (found " +
                         std::to_string(positives.size()) + " positive, " + std::to_string(negatives.size()) + " negative).";
        return CascadeClassifier(config.windowSize, config.minSigma);
    }
    return trainFromImages(positives, negatives, listener, report);
}

std::vector<std::string> Trainer::describe(const TrainReport& report) {
    std::vector<std::string> lines;
    char buffer[160];

    lines.push_back(report.success ? "Training: SUCCESS" : (report.cancelled ? "Training: CANCELLED" : "Training: FAILED"));
    if (!report.message.empty()) {
        lines.push_back(report.message);
    }
    std::snprintf(buffer, sizeof(buffer), "Faces: %d train + %d validation | negative pool: %d | features: %d",
                  report.trainFaces, report.validationFaces, report.negativePool, report.featureCount);
    lines.push_back(buffer);
    for (size_t s = 0; s < report.stages.size(); s++) {
        const StageReport& st = report.stages[s];
        std::snprintf(buffer, sizeof(buffer), "Stage %d: %d weak, threshold %.2f, faces kept %.1f%%, hard negatives %d -> %d",
                      static_cast<int>(s) + 1, st.weakCount, st.threshold, st.validationDetection * 100.0,
                      st.poolBefore, st.poolAfter);
        std::string stageLine = buffer;
        if (st.mined > 0) {
            stageLine += " (" + std::to_string(st.mined) + " mined)";
        }
        lines.push_back(stageLine);
    }
    if (!report.stages.empty()) {
        std::snprintf(buffer, sizeof(buffer), "Whole cascade: faces kept %.1f%%, negative windows passing %.1f%%",
                      report.finalDetection * 100.0, report.finalPoolFalsePositive * 100.0);
        lines.push_back(buffer);
    }
    std::snprintf(buffer, sizeof(buffer), "Time: %.1f s", report.seconds);
    lines.push_back(buffer);
    return lines;
}
