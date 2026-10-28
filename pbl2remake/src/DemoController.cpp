#include "DemoController.h"

#include <algorithm>
#include <cstdio>

#include "Visualizer.h"

namespace {

std::string format(const char* pattern, double a = 0, double b = 0, double c = 0, double d = 0) {
    char buffer[200];
    std::snprintf(buffer, sizeof(buffer), pattern, a, b, c, d);
    return buffer;
}

std::string whole(long long value) {
    return std::to_string(value);
}

}  // namespace

DemoController::DemoController()
    : prepared(false), step(0), original(), gray(), integralVisual(), windowBox(), faceFound(false),
      window(), windowIntegral(), windowSigma(0.0), feature(), weakTrace(), firstStageTrace(),
      cascadeTrace(), windowSize(24), minSigma(4.0), finalBoxes() {
    weakTrace.rawValue = 0;
    weakTrace.sigma = 0.0;
    weakTrace.featureValue = 0.0;
    weakTrace.threshold = 0.0;
    weakTrace.polarity = 1;
    weakTrace.vote = 0;
    weakTrace.alpha = 0.0;
    firstStageTrace.score = 0.0;
    firstStageTrace.sumAlpha = 0.0;
    firstStageTrace.threshold = 0.0;
    firstStageTrace.passed = false;
}

bool DemoController::prepare(const Image& image, const FaceDetector& detector) {
    prepared = false;
    step = 0;
    if (image.isEmpty() || !detector.hasModel()) {
        return false;
    }
    const CascadeClassifier& cascade = detector.getCascade();
    if (cascade.getStage(0).size() == 0) {
        return false;
    }

    original = image;
    gray = image.toGrayscale();
    integralVisual = IntegralImage(gray).toVisualImage();
    windowSize = cascade.getWindowSize();
    minSigma = cascade.getMinSigma();

    // Chọn cửa sổ minh họa: khuôn mặt lớn nhất, hoặc cửa sổ ở giữa ảnh
    finalBoxes = detector.detect(image);
    faceFound = !finalBoxes.empty();
    if (faceFound) {
        windowBox = finalBoxes[0];
        for (size_t i = 1; i < finalBoxes.size(); i++) {
            if (finalBoxes[i].w * finalBoxes[i].h > windowBox.w * windowBox.h) {
                windowBox = finalBoxes[i];
            }
        }
    } else {
        int side = std::max(windowSize, std::min(gray.getWidth(), gray.getHeight()) * 6 / 10);
        windowBox = Rect((gray.getWidth() - side) / 2, (gray.getHeight() - side) / 2, side, side);
    }

    // Đưa khung vào trong ảnh
    int left = std::max(0, windowBox.x);
    int top = std::max(0, windowBox.y);
    int right = std::min(gray.getWidth(), windowBox.x + windowBox.w);
    int bottom = std::min(gray.getHeight(), windowBox.y + windowBox.h);
    windowBox = Rect(left, top, right - left, bottom - top);
    if (windowBox.w < 1 || windowBox.h < 1) {
        return false;
    }

    // Thu cửa sổ về windowSize x windowSize và cho đi qua cascade thật
    Image cropped = gray.crop(windowBox);
    window = (cropped.getWidth() == windowSize && cropped.getHeight() == windowSize)
                 ? cropped : cropped.resized(windowSize, windowSize);
    windowIntegral = IntegralImage(window);
    windowSigma = windowIntegral.windowStdDev(Rect(0, 0, windowSize, windowSize));
    cascade.classifyWindow(windowIntegral, 0, 0, &cascadeTrace);

    const StrongClassifier& firstStage = cascade.getStage(0);
    feature = firstStage.getWeak(0).getFeature();
    weakTrace = firstStage.getWeak(0).explain(windowIntegral, 0, 0, windowSigma);
    firstStageTrace = firstStage.explain(windowIntegral, 0, 0, windowSigma);

    prepared = true;
    return true;
}

bool DemoController::isPrepared() const { return prepared; }
void DemoController::reset() { step = 0; }
int DemoController::getStep() const { return step; }

bool DemoController::next() {
    if (!prepared || step >= STEP_COUNT - 1) {
        return false;
    }
    step++;
    return true;
}

bool DemoController::previous() {
    if (!prepared || step <= 0) {
        return false;
    }
    step--;
    return true;
}

std::string DemoController::getTitle() const {
    static const char* names[STEP_COUNT] = {
        "Original image", "Grayscale", "Integral Image", "Choose a Haar feature",
        "Compute the feature value", "Weak classifier", "Strong classifier (stage 1)",
        "Cascade", "Detection result"
    };
    if (!prepared) {
        return "Demo mode";
    }
    return "Step " + std::to_string(step + 1) + "/" + std::to_string(STEP_COUNT) + ": " + names[step];
}

std::vector<std::string> DemoController::getLines() const {
    std::vector<std::string> lines;
    if (!prepared) {
        lines.push_back("Open an image and load a model,");
        lines.push_back("then press Demo Mode.");
        return lines;
    }

    switch (step) {
    case 0: {
        lines.push_back("Image size: " + std::to_string(original.getWidth()) + " x " + std::to_string(original.getHeight()));
        lines.push_back("Window (yellow): x=" + std::to_string(windowBox.x) + " y=" + std::to_string(windowBox.y) +
                        ", " + std::to_string(windowBox.w) + "x" + std::to_string(windowBox.h));
        lines.push_back(faceFound ? "A face was detected here." : "No face found: using the image center.");
        lines.push_back("It is scaled to " + std::to_string(windowSize) + "x" + std::to_string(windowSize) +
                        " and sent through");
        lines.push_back("the real pipeline.");
        break;
    }
    case 1: {
        lines.push_back("gray = 0.299*R + 0.587*G + 0.114*B");
        lines.push_back("Integer form: (299R+587G+114B+500)/1000");
        if (original.getChannels() == 3) {
            int cx = windowBox.x + windowBox.w / 2;
            int cy = windowBox.y + windowBox.h / 2;
            int r = original.getPixel(cx, cy, 0), g = original.getPixel(cx, cy, 1), b = original.getPixel(cx, cy, 2);
            lines.push_back("Pixel (" + std::to_string(cx) + "," + std::to_string(cy) + "): R=" + std::to_string(r) +
                            " G=" + std::to_string(g) + " B=" + std::to_string(b));
            lines.push_back("-> gray = " + std::to_string(static_cast<int>(gray.getPixel(cx, cy))));
        }
        lines.push_back("Haar features compare light and dark,");
        lines.push_back("so color is not needed.");
        break;
    }
    case 2: {
        lines.push_back("S(x,y) = sum of pixels above and left.");
        lines.push_back("Brighter = bigger sum (grows to the");
        lines.push_back("bottom-right).");
        lines.push_back("Rectangle sum = D - B - C + A");
        lines.push_back("(only 4 table lookups)");
        std::vector<Rect> whites = feature.getWhiteRects();
        if (!whites.empty()) {
            const Rect& r = whites[0];
            long long A = windowIntegral.getValue(r.x, r.y);
            long long B = windowIntegral.getValue(r.x + r.w, r.y);
            long long C = windowIntegral.getValue(r.x, r.y + r.h);
            long long D = windowIntegral.getValue(r.x + r.w, r.y + r.h);
            lines.push_back("Example: first white cell of the feature");
            lines.push_back("  (x=" + std::to_string(r.x) + ",y=" + std::to_string(r.y) + ",w=" + std::to_string(r.w) +
                            ",h=" + std::to_string(r.h) + ") in the window");
            lines.push_back("  A=" + whole(A) + " B=" + whole(B) + " C=" + whole(C) + " D=" + whole(D));
            lines.push_back("  D-B-C+A = " + whole(D - B - C + A));
        }
        break;
    }
    case 3: {
        lines.push_back("Feature: " + feature.getTypeName());
        lines.push_back("Position in window: x=" + std::to_string(feature.getX()) + " y=" + std::to_string(feature.getY()));
        lines.push_back("Cell size: " + std::to_string(feature.getW()) + "x" + std::to_string(feature.getH()) +
                        ", total " + std::to_string(feature.getTotalWidth()) + "x" + std::to_string(feature.getTotalHeight()));
        lines.push_back("Green = WHITE cells (added)");
        lines.push_back("Red   = BLACK cells (subtracted)");
        if (feature.getBlackWeight() == 2) {
            lines.push_back("3-rectangle feature: black sum x 2");
            lines.push_back("(so a flat area gives exactly 0)");
        }
        lines.push_back("First weak classifier of stage 1.");
        break;
    }
    case 4: {
        long long whiteSum = 0, blackSum = 0;
        std::vector<Rect> whites = feature.getWhiteRects();
        std::vector<Rect> blacks = feature.getBlackRects();
        for (size_t i = 0; i < whites.size(); i++) whiteSum += windowIntegral.sumRect(whites[i]);
        for (size_t i = 0; i < blacks.size(); i++) blackSum += windowIntegral.sumRect(blacks[i]);
        lines.push_back("White sum = " + whole(whiteSum));
        lines.push_back("Black sum = " + whole(blackSum));
        lines.push_back("value = white - " + std::to_string(feature.getBlackWeight()) + " * black = " + whole(weakTrace.rawValue));
        lines.push_back(format("Window sigma = %.2f (std deviation)", windowSigma));
        lines.push_back(format("Normalized = value / sigma = %.3f", weakTrace.featureValue));
        lines.push_back("Dividing by sigma removes the effect of");
        lines.push_back("brightness and contrast.");
        break;
    }
    case 5: {
        lines.push_back(format("Threshold = %.3f", weakTrace.threshold));
        lines.push_back("Polarity = " + std::string(weakTrace.polarity > 0 ? "+1: face if value < threshold" : "-1: face if value > threshold"));
        lines.push_back(format("Rule: %.0f * value < %.0f * threshold ?", weakTrace.polarity, weakTrace.polarity));
        lines.push_back(format("%.3f < %.3f ?", weakTrace.polarity * weakTrace.featureValue, weakTrace.polarity * weakTrace.threshold));
        lines.push_back(std::string("Vote = ") + (weakTrace.vote == 1 ? "1  (looks like a FACE)" : "0  (not a face)"));
        lines.push_back(format("Weight alpha = %.3f", weakTrace.alpha));
        lines.push_back(format("Weighted vote = alpha * vote = %.3f", weakTrace.alpha * weakTrace.vote));
        break;
    }
    case 6: {
        int shown = std::min<int>(8, static_cast<int>(firstStageTrace.weaks.size()));
        for (int i = 0; i < shown; i++) {
            const WeakTrace& w = firstStageTrace.weaks[i];
            lines.push_back("Weak " + std::to_string(i + 1) + ": vote " + std::to_string(w.vote) +
                            format(" x alpha %.2f", w.alpha));
        }
        if (static_cast<int>(firstStageTrace.weaks.size()) > shown) {
            lines.push_back("... (" + std::to_string(firstStageTrace.weaks.size() - shown) + " more)");
        }
        lines.push_back(format("Score = sum(alpha * vote) = %.3f", firstStageTrace.score));
        lines.push_back(format("Threshold (stage 1)      = %.3f", firstStageTrace.threshold));
        lines.push_back(std::string("Stage 1: ") + (firstStageTrace.passed ? "PASS" : "REJECT"));
        break;
    }
    case 7: {
        if (cascadeTrace.flatRejected) {
            lines.push_back(format("Window sigma %.2f is below the minimum", cascadeTrace.sigma));
            lines.push_back(format("(%.1f): rejected before stage 1.", minSigma));
        }
        for (size_t s = 0; s < cascadeTrace.stages.size(); s++) {
            const StageTrace& st = cascadeTrace.stages[s];
            lines.push_back("Stage " + std::to_string(s + 1) + format(": score %.2f vs %.2f -> ", st.score, st.threshold) +
                            (st.passed ? "PASS" : "REJECT"));
        }
        if (cascadeTrace.accepted) {
            lines.push_back("Passed ALL stages -> FACE");
        } else if (cascadeTrace.rejectedAtStage >= 0) {
            lines.push_back("Rejected at stage " + std::to_string(cascadeTrace.rejectedAtStage + 1) + " -> NOT a face");
            lines.push_back("(later stages are not even run)");
        }
        break;
    }
    default: {
        lines.push_back("Faces detected: " + std::to_string(finalBoxes.size()));
        for (size_t i = 0; i < finalBoxes.size() && i < 6; i++) {
            lines.push_back("  #" + std::to_string(i + 1) + ": x=" + std::to_string(finalBoxes[i].x) + " y=" + std::to_string(finalBoxes[i].y) +
                            " size " + std::to_string(finalBoxes[i].w) + "x" + std::to_string(finalBoxes[i].h));
        }
        lines.push_back("Whole image: pyramid + sliding window,");
        lines.push_back("cascade; overlapping boxes are merged.");
        break;
    }
    }
    return lines;
}

Image DemoController::render(int areaW, int areaH) const {
    if (!prepared) {
        return Image(areaW, areaH, 3, 30);
    }
    switch (step) {
    case 0: {
        std::vector<Rect> box(1, windowBox);
        return Visualizer::renderBoxes(original, box, areaW, areaH, 255, 220, 0);
    }
    case 1:
        return Visualizer::fitInto(gray, areaW, areaH);
    case 2:
        return Visualizer::fitInto(integralVisual, areaW, areaH);
    case 3:
    case 4:
    case 5:
        return Visualizer::renderFeature(window, feature, areaW, areaH);
    default:
        break;
    }
    if (step == 8) {
        return Visualizer::renderBoxes(original, finalBoxes, areaW, areaH, 0, 255, 0);
    }
    // bước 7, 8: cửa sổ 24x24 viền xanh nếu qua hết, đỏ nếu bị loại
    std::vector<Rect> frame(1, Rect(0, 0, windowSize, windowSize));
    if (cascadeTrace.accepted) {
        return Visualizer::renderBoxes(window, frame, areaW, areaH, 0, 255, 0);
    }
    return Visualizer::renderBoxes(window, frame, areaW, areaH, 255, 0, 0);
}
