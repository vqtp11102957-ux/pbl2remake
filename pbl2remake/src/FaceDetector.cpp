#include "FaceDetector.h"

#include <chrono>
#include <cmath>
#include <fstream>

#include "RectUtils.h"

// ------------------------------------------------------------
// Constructor, model
// ------------------------------------------------------------

FaceDetector::FaceDetector() : cascade(), params() {
}

FaceDetector::FaceDetector(const CascadeClassifier& cascade_) : cascade(cascade_), params() {
}

void FaceDetector::setCascade(const CascadeClassifier& newCascade) { cascade = newCascade; }
const CascadeClassifier& FaceDetector::getCascade() const { return cascade; }
bool FaceDetector::hasModel() const { return cascade.stageCount() > 0; }

bool FaceDetector::loadModel(const std::string& path) {
    std::ifstream file(path.c_str());
    if (!file.is_open()) {
        return false;
    }
    CascadeClassifier loaded;
    if (!loaded.load(file)) {
        return false;
    }
    cascade = loaded;
    return true;
}

bool FaceDetector::saveModel(const std::string& path) const {
    std::ofstream file(path.c_str());
    if (!file.is_open()) {
        return false;
    }
    cascade.save(file);
    return file.good();
}

const DetectorParams& FaceDetector::getParams() const { return params; }

void FaceDetector::setParams(const DetectorParams& newParams) {
    params = newParams;
    if (params.scaleFactor < 1.05) params.scaleFactor = 1.05;
    if (params.step < 1) params.step = 1;
    if (params.minNeighbors < 1) params.minNeighbors = 1;
    if (params.minFaceSize < 1) params.minFaceSize = 1;
}

// ------------------------------------------------------------
// CỬA SỔ TRƯỢT trên một mức kim tự tháp
//
//   y = 0, step, 2*step, ...        (hàng)
//     x = 0, step, 2*step, ...      (cột)
//        cửa sổ 24x24 tại (x, y) --> cascade --> có phải mặt?
//
// Cửa sổ nào là mặt thì ghi lại khung của nó, đổi về tọa độ ảnh gốc.
// ------------------------------------------------------------
void FaceDetector::scanLevel(const IntegralImage& ii, double sx, double sy,
                             std::vector<Rect>& raw, long long& windowCount) const {
    int windowSize = cascade.getWindowSize();
    for (int y = 0; y + windowSize <= ii.getHeight(); y += params.step) {
        for (int x = 0; x + windowSize <= ii.getWidth(); x += params.step) {
            windowCount++;
            if (cascade.classifyWindow(ii, x, y)) {
                int boxX = static_cast<int>(std::floor(x * sx + 0.5));
                int boxY = static_cast<int>(std::floor(y * sy + 0.5));
                int boxW = static_cast<int>(std::floor(windowSize * sx + 0.5));
                int boxH = static_cast<int>(std::floor(windowSize * sy + 0.5));
                raw.push_back(Rect(boxX, boxY, boxW, boxH));
            }
        }
    }
}

// ------------------------------------------------------------
// KIM TỰ THÁP ẢNH: lần lượt thu nhỏ ảnh xám theo scaleFactor, mỗi mức quét cửa sổ trượt.
//   mức 0: ảnh gốc (scale = 1), mức 1: thu nhỏ 1.25 lần, mức 2: 1.25^2 lần, ...
//   dừng khi ảnh nhỏ hơn cửa sổ.
// `factor` là tỉ lệ đã thu nhỏ trước đó (ảnh quá lớn) để khung trả về vẫn đúng trên ảnh gốc.
// ------------------------------------------------------------
void FaceDetector::runPyramid(const Image& gray, double factor, std::vector<Rect>& raw,
                              long long& windowCount, int& levelCount) const {
    int windowSize = cascade.getWindowSize();

    for (double scale = 1.0; ; scale *= params.scaleFactor) {
        int levelW = static_cast<int>(std::floor(gray.getWidth() / scale + 0.5));
        int levelH = static_cast<int>(std::floor(gray.getHeight() / scale + 0.5));
        if (levelW < windowSize || levelH < windowSize) {
            break;                           // ảnh đã nhỏ hơn cửa sổ
        }

        // Mặt mà mức này tìm được rộng khoảng windowSize * scale * factor pixel (trên ảnh gốc)
        double faceSize = windowSize * scale * factor;
        bool tooSmall = faceSize < params.minFaceSize;
        bool tooBig = params.maxFaceSize > 0 && faceSize > params.maxFaceSize;
        if (tooBig) {
            break;                           // các mức sau còn lớn hơn nữa
        }
        if (tooSmall) {
            continue;
        }

        bool sameSize = (levelW == gray.getWidth() && levelH == gray.getHeight());
        Image level = sameSize ? gray : gray.resized(levelW, levelH);
        IntegralImage ii(level);             // MỘT Integral Image cho cả mức này
        double sx = static_cast<double>(gray.getWidth()) / levelW * factor;
        double sy = static_cast<double>(gray.getHeight()) / levelH * factor;
        scanLevel(ii, sx, sy, raw, windowCount);
        levelCount++;
    }
}

std::vector<Rect> FaceDetector::collectRaw(const Image& image, long long& windowCount, int& levelCount) const {
    std::vector<Rect> raw;
    windowCount = 0;
    levelCount = 0;
    if (image.isEmpty() || !hasModel()) {
        return raw;
    }

    Image gray = image.toGrayscale();
    double factor = 1.0;
    if (params.maxProcessingWidth > 0 && gray.getWidth() > params.maxProcessingWidth) {
        int newWidth = params.maxProcessingWidth;
        int newHeight = static_cast<int>(std::floor(gray.getHeight() * static_cast<double>(newWidth) / gray.getWidth() + 0.5));
        if (newHeight < 1) {
            newHeight = 1;
        }
        factor = static_cast<double>(gray.getWidth()) / newWidth;
        gray = gray.resized(newWidth, newHeight);
    }
    runPyramid(gray, factor, raw, windowCount, levelCount);
    return raw;
}

std::vector<Rect> FaceDetector::detectRaw(const Image& image) const {
    long long windows = 0;
    int levels = 0;
    return collectRaw(image, windows, levels);
}

// ------------------------------------------------------------
// Gộp khung chồng nhau
//
// Một khuôn mặt thật thường làm nhiều cửa sổ lân cận (và nhiều mức tỉ lệ) cùng "dương".
// Ta gom các khung chồng nhau thành NHÓM (union-find), bỏ các nhóm quá ít khung
// (thường là nhận nhầm lẻ tẻ), rồi thay mỗi nhóm bằng khung trung bình.
// ------------------------------------------------------------

static int findRoot(std::vector<int>& parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];       // rút gọn đường đi
        i = parent[i];
    }
    return i;
}

std::vector<Rect> FaceDetector::mergeDetections(const std::vector<Rect>& raw, double overlap, int minNeighbors) {
    int n = static_cast<int>(raw.size());
    std::vector<int> parent(n);
    for (int i = 0; i < n; i++) {
        parent[i] = i;
    }

    // Hai khung chồng nhau nhiều thì cùng nhóm
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (overlapRatio(raw[i], raw[j]) > overlap) {
                int a = findRoot(parent, i);
                int b = findRoot(parent, j);
                if (a != b) {
                    parent[b] = a;
                }
            }
        }
    }

    // Cộng dồn từng nhóm
    std::vector<double> sumX(n, 0.0), sumY(n, 0.0), sumW(n, 0.0), sumH(n, 0.0);
    std::vector<int> count(n, 0);
    for (int i = 0; i < n; i++) {
        int root = findRoot(parent, i);
        sumX[root] += raw[i].x;
        sumY[root] += raw[i].y;
        sumW[root] += raw[i].w;
        sumH[root] += raw[i].h;
        count[root]++;
    }

    // Nhóm đủ lớn -> khung trung bình
    std::vector<Rect> boxes;
    std::vector<int> boxCounts;
    for (int r = 0; r < n; r++) {
        if (count[r] > 0 && count[r] >= minNeighbors) {
            double c = count[r];
            boxes.push_back(Rect(static_cast<int>(std::floor(sumX[r] / c + 0.5)),
                                 static_cast<int>(std::floor(sumY[r] / c + 0.5)),
                                 static_cast<int>(std::floor(sumW[r] / c + 0.5)),
                                 static_cast<int>(std::floor(sumH[r] / c + 0.5))));
            boxCounts.push_back(count[r]);
        }
    }

    // Bỏ khung nhỏ nằm gần hết bên trong một khung lớn hơn có nhiều phiếu hơn (hoặc bằng)
    std::vector<Rect> result;
    for (size_t i = 0; i < boxes.size(); i++) {
        bool nested = false;
        int areaI = boxes[i].w * boxes[i].h;
        for (size_t j = 0; j < boxes.size() && !nested; j++) {
            if (i == j) {
                continue;
            }
            int areaJ = boxes[j].w * boxes[j].h;
            bool insideMostly = intersectionArea(boxes[i], boxes[j]) > 0.8 * areaI;
            if (insideMostly && areaJ > areaI && boxCounts[j] >= boxCounts[i]) {
                nested = true;
            }
        }
        if (!nested) {
            result.push_back(boxes[i]);
        }
    }
    return result;
}

// ------------------------------------------------------------
// Phát hiện đầy đủ
// ------------------------------------------------------------
std::vector<Rect> FaceDetector::detect(const Image& image, DetectionStats* stats) const {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    long long windows = 0;
    int levels = 0;
    std::vector<Rect> raw = collectRaw(image, windows, levels);
    std::vector<Rect> merged = mergeDetections(raw, params.mergeOverlap, params.minNeighbors);

    if (stats != 0) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        stats->levels = levels;
        stats->windowsEvaluated = windows;
        stats->rawDetections = static_cast<int>(raw.size());
        stats->mergedDetections = static_cast<int>(merged.size());
        stats->milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return merged;
}
