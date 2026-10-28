#include "AdaBoost.h"

#include <cmath>
#include <string>

// Lỗi nhỏ nhất được phép dùng để tính alpha (tránh chia cho 0 / log(vô cực)).
static const double MIN_ERROR = 1e-10;

// Nếu lỗi tốt nhất >= 0.5 - STOP_MARGIN thì coi như không còn feature nào có ích.
static const double STOP_MARGIN = 1e-9;

AdaBoost::AdaBoost(int rounds_) : rounds(rounds_) {
}

int AdaBoost::getRounds() const { return rounds; }

// ------------------------------------------------------------
// Trọng số
// ------------------------------------------------------------

std::vector<double> AdaBoost::initialWeights(const std::vector<Label>& labels) {
    int faces = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == FACE) faces++;
    }
    int nonFaces = static_cast<int>(labels.size()) - faces;

    std::vector<double> weights(labels.size(), 0.0);
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == FACE) {
            weights[i] = 1.0 / (2.0 * faces);
        } else {
            weights[i] = 1.0 / (2.0 * nonFaces);
        }
    }
    return weights;
}

void AdaBoost::normalizeWeights(std::vector<double>& weights) {
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        sum += weights[i];
    }
    if (sum <= 0.0) {
        return;
    }
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] /= sum;
    }
}

// ------------------------------------------------------------
// Chuẩn bị: tính trước giá trị feature và thứ tự sắp xếp
// ------------------------------------------------------------

// values[f][i] = giá trị đã chuẩn hóa của feature f trên mẫu i.
// Giá trị này KHÔNG phụ thuộc trọng số nên chỉ cần tính một lần cho mọi vòng.
bool AdaBoost::computeFeatureValues(const TrainingSet& data,
                                    const std::vector<HaarFeature>& features,
                                    std::vector<std::vector<double> >& values,
                                    ProgressListener* listener) {
    int n = data.size();
    values.assign(features.size(), std::vector<double>(n, 0.0));
    for (size_t f = 0; f < features.size(); f++) {
        for (int i = 0; i < n; i++) {
            long long raw = features[f].compute(data.getImage(i));
            values[f][i] = WeakClassifier::normalize(raw, data.getSigma(i));
        }
        if (listener != 0 && f % 128 == 0) {
            int percent = static_cast<int>(30 * f / features.size());
            listener->onProgress("Computing Haar feature values", percent);
            if (listener->shouldCancel()) {
                return false;
            }
        }
    }
    return true;
}

// orders[f] = các chỉ số mẫu xếp theo giá trị feature f tăng dần.
// Thứ tự này cũng KHÔNG phụ thuộc trọng số nên sắp xếp một lần.
void AdaBoost::computeSortedOrders(const std::vector<std::vector<double> >& values,
                                   std::vector<std::vector<int> >& orders) {
    orders.resize(values.size());
    for (size_t f = 0; f < values.size(); f++) {
        orders[f] = WeakClassifier::sortedOrder(values[f]);
    }
}

// ------------------------------------------------------------
// Một vòng: chọn feature tốt nhất với trọng số hiện tại
// ------------------------------------------------------------

int AdaBoost::findBestFeature(const std::vector<std::vector<double> >& values,
                              const std::vector<std::vector<int> >& orders,
                              const std::vector<Label>& labels,
                              const std::vector<double>& weights,
                              ThresholdResult& bestResult) {
    int bestIndex = 0;
    bool first = true;
    for (size_t f = 0; f < values.size(); f++) {
        ThresholdResult r = WeakClassifier::findBestThreshold(values[f], orders[f], labels, weights);
        if (first || r.error < bestResult.error) {
            bestResult = r;
            bestIndex = static_cast<int>(f);
            first = false;
        }
    }
    return bestIndex;
}

// ------------------------------------------------------------
// Vòng lặp AdaBoost
// ------------------------------------------------------------

AdaBoostResult AdaBoost::train(const TrainingSet& data, const std::vector<HaarFeature>& features,
                               ProgressListener* listener) const {
    AdaBoostResult result;
    int n = data.size();
    if (rounds < 1 || features.empty() || data.countFaces() == 0 || data.countNonFaces() == 0) {
        return result;
    }

    // Chuẩn bị (làm một lần)
    std::vector<std::vector<double> > values;
    std::vector<std::vector<int> > orders;
    if (!computeFeatureValues(data, features, values, listener)) {
        result.cancelled = true;
        return result;
    }
    if (listener != 0) {
        listener->onProgress("Sorting feature values", 30);
    }
    computeSortedOrders(values, orders);

    const std::vector<Label>& labels = data.getLabels();
    int faceCount = data.countFaces();
    int nonFaceCount = data.countNonFaces();

    // Bước 1: khởi tạo trọng số
    std::vector<double> weights = initialWeights(labels);

    std::vector<double> score(n, 0.0);   // sum(alpha_t * vote_t) của từng mẫu
    double sumAlpha = 0.0;

    for (int t = 1; t <= rounds; t++) {
        if (listener != 0) {
            listener->onProgress("AdaBoost round " + std::to_string(t) + "/" + std::to_string(rounds),
                                 35 + 65 * (t - 1) / rounds);
            if (listener->shouldCancel()) {
                result.cancelled = true;
                return result;
            }
        }

        // Bước 2-3: chuẩn hóa trọng số, chọn weak classifier tốt nhất
        normalizeWeights(weights);
        ThresholdResult best;
        int featureIndex = findBestFeature(values, orders, labels, weights, best);
        if (best.error >= 0.5 - STOP_MARGIN) {
            break;                       // không còn feature nào hơn đoán ngẫu nhiên
        }

        // Bước 4: alpha
        double error = (best.error < MIN_ERROR) ? MIN_ERROR : best.error;
        double alpha = std::log((1.0 - error) / error);
        double beta = error / (1.0 - error);
        WeakClassifier weak(features[featureIndex], best.threshold, best.polarity, alpha);

        // Bước 5: mẫu đoán đúng được nhân beta (<1), mẫu đoán sai giữ nguyên
        std::vector<int> votes(n);
        for (int i = 0; i < n; i++) {
            votes[i] = weak.voteOnValue(values[featureIndex][i]);
            bool predictedFace = (votes[i] == 1);
            bool isFace = (labels[i] == FACE);
            if (predictedFace == isFace) {
                weights[i] *= beta;
            }
        }
        normalizeWeights(weights);

        // Ghi lại: lỗi của weak classifier này với trọng số mới
        double errorAfter = 0.0;
        for (int i = 0; i < n; i++) {
            if ((votes[i] == 1) != (labels[i] == FACE)) {
                errorAfter += weights[i];
            }
        }

        // Strong classifier hiện tại (t vòng đầu): FACE nếu score >= 0.5 * sumAlpha
        sumAlpha += alpha;
        int wrong = 0, facesFound = 0, nonFacesFound = 0;
        for (int i = 0; i < n; i++) {
            score[i] += alpha * votes[i];
            bool predictedFace = (score[i] >= 0.5 * sumAlpha);
            bool isFace = (labels[i] == FACE);
            if (predictedFace != isFace) wrong++;
            if (predictedFace && isFace) facesFound++;
            if (predictedFace && !isFace) nonFacesFound++;
        }

        RoundInfo info;
        info.round = t;
        info.featureIndex = featureIndex;
        info.error = best.error;
        info.alpha = alpha;
        info.errorAfterUpdate = errorAfter;
        info.ensembleError = static_cast<double>(wrong) / n;
        info.detectionRate = static_cast<double>(facesFound) / faceCount;
        info.falsePositiveRate = static_cast<double>(nonFacesFound) / nonFaceCount;

        result.weakClassifiers.push_back(weak);
        result.history.push_back(info);
    }
    return result;
}
