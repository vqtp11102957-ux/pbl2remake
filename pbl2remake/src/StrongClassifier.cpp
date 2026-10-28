#include "StrongClassifier.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

StrongClassifier::StrongClassifier() : weaks(), threshold(0.0) {
}

StrongClassifier::StrongClassifier(const std::vector<WeakClassifier>& weakClassifiers)
    : weaks(weakClassifiers), threshold(0.0) {
    threshold = getDefaultThreshold();
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

int StrongClassifier::size() const { return static_cast<int>(weaks.size()); }
const WeakClassifier& StrongClassifier::getWeak(int index) const { return weaks[index]; }

double StrongClassifier::getSumAlpha() const {
    double sum = 0.0;
    for (size_t i = 0; i < weaks.size(); i++) {
        sum += weaks[i].getAlpha();
    }
    return sum;
}

double StrongClassifier::getDefaultThreshold() const { return 0.5 * getSumAlpha(); }
double StrongClassifier::getThreshold() const { return threshold; }
void StrongClassifier::setThreshold(double newThreshold) { threshold = newThreshold; }

// ------------------------------------------------------------
// Chấm điểm
// ------------------------------------------------------------

// score = tổng (alpha * phiếu) của mọi weak classifier.
double StrongClassifier::score(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const {
    double total = 0.0;
    for (size_t i = 0; i < weaks.size(); i++) {
        total += weaks[i].getAlpha() * weaks[i].predict(ii, offsetX, offsetY, sigma);
    }
    return total;
}

bool StrongClassifier::predict(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const {
    return score(ii, offsetX, offsetY, sigma) >= threshold;
}

StageTrace StrongClassifier::explain(const IntegralImage& ii, int offsetX, int offsetY, double sigma) const {
    StageTrace trace;
    trace.score = 0.0;
    for (size_t i = 0; i < weaks.size(); i++) {
        WeakTrace w = weaks[i].explain(ii, offsetX, offsetY, sigma);
        trace.score += w.alpha * w.vote;
        trace.weaks.push_back(w);
    }
    trace.sumAlpha = getSumAlpha();
    trace.threshold = threshold;
    trace.passed = (trace.score >= threshold);
    return trace;
}

// ------------------------------------------------------------
// Đánh giá trên tập mẫu
// ------------------------------------------------------------

RateStats StrongClassifier::computeRates(const TrainingSet& data) const {
    RateStats stats;
    stats.faceCount = 0;
    stats.nonFaceCount = 0;
    stats.faceAccepted = 0;
    stats.nonFaceAccepted = 0;

    for (int i = 0; i < data.size(); i++) {
        bool accepted = predict(data.getImage(i), 0, 0, data.getSigma(i));
        if (data.getLabel(i) == FACE) {
            stats.faceCount++;
            if (accepted) stats.faceAccepted++;
        } else {
            stats.nonFaceCount++;
            if (accepted) stats.nonFaceAccepted++;
        }
    }

    stats.detectionRate = (stats.faceCount > 0) ? static_cast<double>(stats.faceAccepted) / stats.faceCount : 0.0;
    stats.falsePositiveRate = (stats.nonFaceCount > 0) ? static_cast<double>(stats.nonFaceAccepted) / stats.nonFaceCount : 0.0;
    int total = stats.faceCount + stats.nonFaceCount;
    int wrong = (stats.faceCount - stats.faceAccepted) + stats.nonFaceAccepted;
    stats.errorRate = (total > 0) ? static_cast<double>(wrong) / total : 0.0;
    return stats;
}

// ------------------------------------------------------------
// Chọn ngưỡng cao nhất mà vẫn giữ được tỉ lệ mặt yêu cầu.
//
// Tính điểm của mọi mẫu FACE, xếp tăng dần: s[0] <= s[1] <= ... <= s[m-1].
// Cho phép k mặt có điểm thấp nhất bị loại, với k = floor((1 - rate) * m).
// Đặt ngưỡng = s[k]: các mặt có điểm >= s[k] đều đi qua (ít nhất m - k mặt).
// ------------------------------------------------------------
bool StrongClassifier::calibrateThreshold(const TrainingSet& data, double targetDetectionRate) {
    std::vector<double> faceScores;
    for (int i = 0; i < data.size(); i++) {
        if (data.getLabel(i) == FACE) {
            faceScores.push_back(score(data.getImage(i), 0, 0, data.getSigma(i)));
        }
    }
    if (faceScores.empty()) {
        return false;
    }
    std::sort(faceScores.begin(), faceScores.end());

    if (targetDetectionRate < 0.0) targetDetectionRate = 0.0;
    if (targetDetectionRate > 1.0) targetDetectionRate = 1.0;

    int m = static_cast<int>(faceScores.size());
    int k = static_cast<int>(std::floor((1.0 - targetDetectionRate) * m + 1e-9));
    if (k > m - 1) {
        k = m - 1;                     // luôn giữ ít nhất một mặt
    }
    threshold = faceScores[k];
    return true;
}

// ------------------------------------------------------------
// Lưu / đọc
// ------------------------------------------------------------

void StrongClassifier::save(std::ostream& out) const {
    std::streamsize oldPrecision = out.precision();
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "THRESHOLD " << threshold << "\n";
    out << "WEAKS " << weaks.size() << "\n";
    out.precision(oldPrecision);
    for (size_t i = 0; i < weaks.size(); i++) {
        weaks[i].save(out);
    }
}

bool StrongClassifier::load(std::istream& in) {
    std::string keyThreshold, keyWeaks;
    double newThreshold;
    long count;

    in >> keyThreshold >> newThreshold >> keyWeaks >> count;
    bool headerOk = !in.fail() && keyThreshold == "THRESHOLD" && keyWeaks == "WEAKS" &&
                    std::isfinite(newThreshold) && count >= 0 && count <= 100000;
    if (!headerOk) {
        return false;
    }

    std::vector<WeakClassifier> loaded;
    for (long i = 0; i < count; i++) {
        WeakClassifier w;
        if (!w.load(in)) {
            return false;
        }
        loaded.push_back(w);
    }

    weaks = loaded;
    threshold = newThreshold;
    return true;
}
