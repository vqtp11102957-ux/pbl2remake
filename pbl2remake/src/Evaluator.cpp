#include "Evaluator.h"

#include <cstdio>
#include <fstream>
#include <filesystem>

#include "DatasetLoader.h"

void Evaluator::processImage(const FaceDetector& detector, const Image& image, bool hasFace,
                             const std::string& name, EvaluationResult& result) {
    DetectionStats stats;
    std::vector<Rect> faces = detector.detect(image, &stats);
    result.totalMilliseconds += stats.milliseconds;
    result.facesFound += static_cast<int>(faces.size());

    bool found = !faces.empty();
    if (hasFace) {
        result.positiveImages++;
        if (found) {
            result.truePositives++;
        } else {
            result.falseNegatives++;
        }
    } else {
        result.negativeImages++;
        if (found) {
            result.falsePositives++;
        } else {
            result.trueNegatives++;
        }
    }

    bool wrong = (hasFace != found);
    if (wrong && result.mistakes.size() < 20) {
        result.mistakes.push_back(std::string(hasFace ? "missed face: " : "false alarm: ") + name);
    }
}

void Evaluator::finish(EvaluationResult& result) {
    int total = result.totalImages();
    if (total > 0) {
        result.accuracy = static_cast<double>(result.truePositives + result.trueNegatives) / total;
        result.averageMilliseconds = result.totalMilliseconds / total;
    }
    int reportedFace = result.truePositives + result.falsePositives;
    if (reportedFace > 0) {
        result.precision = static_cast<double>(result.truePositives) / reportedFace;
    }
    int realFace = result.truePositives + result.falseNegatives;
    if (realFace > 0) {
        result.recall = static_cast<double>(result.truePositives) / realFace;
    }
}

EvaluationResult Evaluator::evaluateImages(const FaceDetector& detector,
                                           const std::vector<Image>& positives,
                                           const std::vector<Image>& negatives,
                                           ProgressListener* listener) {
    EvaluationResult result;
    int total = static_cast<int>(positives.size() + negatives.size());
    int done = 0;

    for (size_t i = 0; i < positives.size(); i++) {
        processImage(detector, positives[i], true, "positive #" + std::to_string(i + 1), result);
        done++;
        if (listener != 0) {
            listener->onProgress("Evaluating", 100 * done / total);
            if (listener->shouldCancel()) {
                finish(result);
                return result;
            }
        }
    }
    for (size_t i = 0; i < negatives.size(); i++) {
        processImage(detector, negatives[i], false, "negative #" + std::to_string(i + 1), result);
        done++;
        if (listener != 0) {
            listener->onProgress("Evaluating", 100 * done / total);
            if (listener->shouldCancel()) {
                finish(result);
                return result;
            }
        }
    }
    finish(result);
    return result;
}

EvaluationResult Evaluator::evaluateFolders(const FaceDetector& detector, const std::string& testFolder,
                                            ProgressListener* listener) {
    std::filesystem::path root(testFolder);
    std::vector<std::string> positiveFiles = DatasetLoader::listImages((root / "positive").string());
    std::vector<std::string> negativeFiles = DatasetLoader::listImages((root / "negative").string());

    EvaluationResult result;
    int total = static_cast<int>(positiveFiles.size() + negativeFiles.size());
    if (total == 0) {
        return result;
    }

    int done = 0;
    for (int pass = 0; pass < 2; pass++) {
        const std::vector<std::string>& files = (pass == 0) ? positiveFiles : negativeFiles;
        for (size_t i = 0; i < files.size(); i++) {
            Image image;
            if (image.load(files[i])) {                  // file không đọc được thì bỏ qua
                processImage(detector, image, pass == 0,
                             std::filesystem::path(files[i]).filename().string(), result);
            }
            done++;
            if (listener != 0) {
                listener->onProgress("Evaluating", 100 * done / total);
                if (listener->shouldCancel()) {
                    finish(result);
                    return result;
                }
            }
        }
    }
    finish(result);
    return result;
}

static std::string percentText(double value, bool defined) {
    if (!defined) {
        return "n/a";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f%%", value * 100.0);
    return buffer;
}

std::vector<std::string> Evaluator::describe(const EvaluationResult& r) {
    std::vector<std::string> lines;
    char buffer[160];

    if (r.totalImages() == 0) {
        lines.push_back("No test images found.");
        return lines;
    }
    std::snprintf(buffer, sizeof(buffer), "Test images: %d (%d with face, %d without)",
                  r.totalImages(), r.positiveImages, r.negativeImages);
    lines.push_back(buffer);
    std::snprintf(buffer, sizeof(buffer), "Correct: %d face images found (TP) + %d clean images (TN)",
                  r.truePositives, r.trueNegatives);
    lines.push_back(buffer);
    std::snprintf(buffer, sizeof(buffer), "Wrong: %d missed faces (FN) + %d false alarms (FP)",
                  r.falseNegatives, r.falsePositives);
    lines.push_back(buffer);
    lines.push_back("Accuracy : " + percentText(r.accuracy, true));
    lines.push_back("Precision: " + percentText(r.precision, r.truePositives + r.falsePositives > 0));
    lines.push_back("Recall   : " + percentText(r.recall, r.truePositives + r.falseNegatives > 0));
    std::snprintf(buffer, sizeof(buffer), "Faces found: %d | avg %.1f ms/image (total %.0f ms)",
                  r.facesFound, r.averageMilliseconds, r.totalMilliseconds);
    lines.push_back(buffer);
    for (size_t i = 0; i < r.mistakes.size(); i++) {
        lines.push_back("  - " + r.mistakes[i]);
    }
    return lines;
}

bool Evaluator::saveReport(const EvaluationResult& result, const std::string& path) {
    std::ofstream file(path.c_str());
    if (!file.is_open()) {
        return false;
    }
    std::vector<std::string> lines = describe(result);
    for (size_t i = 0; i < lines.size(); i++) {
        file << lines[i] << "\n";
    }
    return file.good();
}
