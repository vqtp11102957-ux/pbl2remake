#include "CascadeClassifier.h"

#include <cmath>
#include <iomanip>
#include <limits>

// ------------------------------------------------------------
// Constructor và xây dựng
// ------------------------------------------------------------

CascadeClassifier::CascadeClassifier(int windowSize_, double minSigma_)
    : windowSize(windowSize_), minSigma(minSigma_), stages() {
    if (windowSize < 1) {
        windowSize = 1;
    }
    if (minSigma < 0.0) {
        minSigma = 0.0;
    }
}

void CascadeClassifier::addStage(const StrongClassifier& stage) {
    stages.push_back(stage);
}

void CascadeClassifier::clear() {
    stages.clear();
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

int CascadeClassifier::getWindowSize() const { return windowSize; }
double CascadeClassifier::getMinSigma() const { return minSigma; }

void CascadeClassifier::setMinSigma(double newMinSigma) {
    minSigma = (newMinSigma < 0.0) ? 0.0 : newMinSigma;
}

int CascadeClassifier::stageCount() const { return static_cast<int>(stages.size()); }
const StrongClassifier& CascadeClassifier::getStage(int index) const { return stages[index]; }

int CascadeClassifier::totalWeakCount() const {
    int total = 0;
    for (size_t i = 0; i < stages.size(); i++) {
        total += stages[i].size();
    }
    return total;
}

std::string CascadeClassifier::summary() const {
    std::string text = std::to_string(stages.size()) + " tang [";
    for (size_t i = 0; i < stages.size(); i++) {
        if (i > 0) {
            text += ", ";
        }
        text += std::to_string(stages[i].size());
    }
    text += "], tong " + std::to_string(totalWeakCount()) + " weak classifier";
    return text;
}

// ------------------------------------------------------------
// Phân loại một cửa sổ
//
//   1. Cửa sổ ngoài ảnh -> loại.
//   2. Tính sigma MỘT lần cho cả cửa sổ. Quá phẳng -> loại ngay.
//   3. Lần lượt qua từng tầng. Tầng nào loại thì DỪNG, không chạy tầng sau.
//   4. Qua hết mọi tầng -> MẶT.
// ------------------------------------------------------------
bool CascadeClassifier::classifyWindow(const IntegralImage& ii, int offsetX, int offsetY,
                                       WindowTrace* trace) const {
    Rect box(offsetX, offsetY, windowSize, windowSize);
    if (trace != 0) {
        *trace = WindowTrace();
        trace->box = box;
    }

    bool inside = offsetX >= 0 && offsetY >= 0 &&
                  offsetX + windowSize <= ii.getWidth() &&
                  offsetY + windowSize <= ii.getHeight();
    if (!inside) {
        return false;
    }

    double sigma = ii.windowStdDev(box);
    if (trace != 0) {
        trace->sigma = sigma;
    }
    if (sigma < minSigma) {
        if (trace != 0) {
            trace->flatRejected = true;
        }
        return false;
    }

    for (size_t s = 0; s < stages.size(); s++) {
        bool passed;
        if (trace != 0) {
            StageTrace stageTrace = stages[s].explain(ii, offsetX, offsetY, sigma);
            passed = stageTrace.passed;
            trace->stages.push_back(stageTrace);
        } else {
            passed = stages[s].predict(ii, offsetX, offsetY, sigma);
        }
        if (!passed) {
            if (trace != 0) {
                trace->rejectedAtStage = static_cast<int>(s);
            }
            return false;
        }
    }

    if (trace != 0) {
        trace->accepted = true;
    }
    return true;
}

// ------------------------------------------------------------
// Lưu / đọc
// ------------------------------------------------------------

void CascadeClassifier::save(std::ostream& out) const {
    std::streamsize oldPrecision = out.precision();
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "VJ_MODEL 1\n";
    out << "WINDOW " << windowSize << "\n";
    out << "MINSIGMA " << minSigma << "\n";
    out << "STAGES " << stages.size() << "\n";
    out.precision(oldPrecision);
    for (size_t s = 0; s < stages.size(); s++) {
        out << "STAGE " << s << "\n";
        stages[s].save(out);
    }
}

bool CascadeClassifier::load(std::istream& in) {
    std::string keyModel, keyWindow, keySigma, keyStages;
    int version, newWindow;
    double newMinSigma;
    long stageTotal;

    in >> keyModel >> version >> keyWindow >> newWindow >> keySigma >> newMinSigma >> keyStages >> stageTotal;
    bool headerOk = !in.fail() && keyModel == "VJ_MODEL" && version == 1 &&
                    keyWindow == "WINDOW" && newWindow >= 1 && newWindow <= 1000 &&
                    keySigma == "MINSIGMA" && std::isfinite(newMinSigma) && newMinSigma >= 0.0 &&
                    keyStages == "STAGES" && stageTotal >= 0 && stageTotal <= 1000;
    if (!headerOk) {
        return false;
    }

    std::vector<StrongClassifier> loaded;
    for (long s = 0; s < stageTotal; s++) {
        std::string keyStage;
        long index;
        in >> keyStage >> index;
        if (in.fail() || keyStage != "STAGE" || index != s) {
            return false;
        }
        StrongClassifier stage;
        if (!stage.load(in)) {
            return false;
        }
        // Mọi feature phải nằm trọn trong cửa sổ, nếu không file bị hỏng / sai kích thước.
        for (int w = 0; w < stage.size(); w++) {
            if (!stage.getWeak(w).getFeature().fitsInWindow(newWindow)) {
                return false;
            }
        }
        loaded.push_back(stage);
    }

    windowSize = newWindow;
    minSigma = newMinSigma;
    stages = loaded;
    return true;
}
