#include "WeakClassifier.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

WeakClassifier::WeakClassifier()
    : feature(), threshold(0.0), polarity(1), alpha(0.0) {
}

WeakClassifier::WeakClassifier(const HaarFeature& feature_, double threshold_,
                               int polarity_, double alpha_)
    : feature(feature_), threshold(threshold_), polarity(polarity_), alpha(alpha_) {
}

// ------------------------------------------------------------
// Thông tin
// ------------------------------------------------------------

const HaarFeature& WeakClassifier::getFeature() const { return feature; }
double WeakClassifier::getThreshold() const { return threshold; }
int WeakClassifier::getPolarity() const { return polarity; }
double WeakClassifier::getAlpha() const { return alpha; }
void WeakClassifier::setAlpha(double newAlpha) { alpha = newAlpha; }

// ------------------------------------------------------------
// Dự đoán
// ------------------------------------------------------------

// Chuẩn hóa độ sáng: chia giá trị thô cho độ lệch chuẩn của cửa sổ.
// Cửa sổ gần như phẳng (sigma < 1) được coi như sigma = 1 để không chia cho 0.
double WeakClassifier::normalize(long long rawValue, double sigma) {
    const double MIN_SIGMA = 1.0;
    if (sigma < MIN_SIGMA) {
        sigma = MIN_SIGMA;
    }
    return static_cast<double>(rawValue) / sigma;
}

double WeakClassifier::featureValue(const IntegralImage& ii, int offsetX, int offsetY,
                                    double sigma) const {
    return normalize(feature.compute(ii, offsetX, offsetY), sigma);
}

// polarity * value < polarity * threshold  ->  mặt
//   polarity = +1:  value <  threshold
//   polarity = -1:  value >  threshold
int WeakClassifier::voteOnValue(double normalizedValue) const {
    if (polarity * normalizedValue < polarity * threshold) {
        return 1;
    }
    return 0;
}

int WeakClassifier::predict(const IntegralImage& ii, int offsetX, int offsetY,
                            double sigma) const {
    return voteOnValue(featureValue(ii, offsetX, offsetY, sigma));
}

WeakTrace WeakClassifier::explain(const IntegralImage& ii, int offsetX, int offsetY,
                                  double sigma) const {
    WeakTrace trace;
    trace.feature = feature;
    trace.rawValue = feature.compute(ii, offsetX, offsetY);
    trace.sigma = sigma;
    trace.featureValue = normalize(trace.rawValue, sigma);
    trace.threshold = threshold;
    trace.polarity = polarity;
    trace.vote = voteOnValue(trace.featureValue);
    trace.alpha = alpha;
    return trace;
}

// ------------------------------------------------------------
// Sắp xếp chỉ số theo giá trị tăng dần
// (hàm so sánh nhỏ viết ngay tại chỗ: so sánh values[a] với values[b])
// stable_sort giữ nguyên thứ tự của các mẫu có giá trị bằng nhau.
// ------------------------------------------------------------
std::vector<int> WeakClassifier::sortedOrder(const std::vector<double>& values) {
    std::vector<int> order(values.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = static_cast<int>(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&values](int a, int b) { return values[a] < values[b]; });
    return order;
}

// ------------------------------------------------------------
// Tìm ngưỡng tốt nhất bằng MỘT lượt quét qua các mẫu đã sắp xếp.
//
// Đặt ngưỡng ở giữa hai mẫu liền kề (sau khi sắp xếp). Gọi "bên trái" là các
// mẫu có giá trị nhỏ hơn ngưỡng, "bên phải" là các mẫu còn lại:
//
//   polarity = +1 (bên trái đoán là mặt):
//       sai = (NON_FACE bên trái) + (FACE bên phải)
//   polarity = -1 (bên phải đoán là mặt):
//       sai = (FACE bên trái) + (NON_FACE bên phải)
//
// Khi quét, ta cộng dồn trọng số FACE / NON_FACE của bên trái, còn bên phải
// suy ra bằng (tổng - bên trái). Nhờ vậy tính được lỗi của MỌI ngưỡng chỉ với
// một vòng lặp, không cần thử từng ngưỡng.
// ------------------------------------------------------------
ThresholdResult WeakClassifier::findBestThreshold(const std::vector<double>& values,
                                                  const std::vector<int>& order,
                                                  const std::vector<Label>& labels,
                                                  const std::vector<double>& weights) {
    ThresholdResult best;
    best.threshold = 0.0;
    best.polarity = 1;
    best.error = std::numeric_limits<double>::max();

    int n = static_cast<int>(order.size());
    if (n == 0) {
        best.error = 0.0;
        return best;
    }

    // Tổng trọng số của mọi mẫu FACE và mọi mẫu NON_FACE.
    double totalFace = 0.0;
    double totalNon = 0.0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == FACE) {
            totalFace += weights[i];
        } else {
            totalNon += weights[i];
        }
    }

    double faceLeft = 0.0;      // tổng trọng số FACE ở bên trái ngưỡng
    double nonLeft = 0.0;       // tổng trọng số NON_FACE ở bên trái ngưỡng

    // k = -1: ngưỡng nhỏ hơn mọi giá trị (bên trái rỗng). k = 0..n-1: bên trái
    // gồm các mẫu order[0..k]. Chỉ được tách giữa hai giá trị KHÁC nhau.
    for (int k = -1; k < n; k++) {
        if (k >= 0) {
            int index = order[k];
            if (labels[index] == FACE) {
                faceLeft += weights[index];
            } else {
                nonLeft += weights[index];
            }
            bool isLast = (k == n - 1);
            if (!isLast && values[order[k]] == values[order[k + 1]]) {
                continue;       // hai mẫu bằng nhau không thể nằm hai phía ngưỡng
            }
        }

        double errorPlus = nonLeft + (totalFace - faceLeft);    // polarity = +1
        double errorMinus = faceLeft + (totalNon - nonLeft);    // polarity = -1
        int polarityHere = (errorPlus <= errorMinus) ? 1 : -1;
        double errorHere = (polarityHere == 1) ? errorPlus : errorMinus;

        if (errorHere < best.error) {
            best.error = errorHere;
            best.polarity = polarityHere;
            if (k == -1) {
                best.threshold = values[order[0]] - 1.0;
            } else if (k == n - 1) {
                best.threshold = values[order[n - 1]] + 1.0;
            } else {
                best.threshold = (values[order[k]] + values[order[k + 1]]) / 2.0;
            }
        }
    }
    return best;
}

// ------------------------------------------------------------
// Lưu / đọc một dòng:  TYPE t X x Y y W w H h THR thr POL p ALPHA a
// ------------------------------------------------------------

void WeakClassifier::save(std::ostream& out) const {
    std::streamsize oldPrecision = out.precision();
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "TYPE " << static_cast<int>(feature.getType())
        << " X " << feature.getX() << " Y " << feature.getY()
        << " W " << feature.getW() << " H " << feature.getH()
        << " THR " << threshold << " POL " << polarity
        << " ALPHA " << alpha << "\n";
    out.precision(oldPrecision);
}

bool WeakClassifier::load(std::istream& in) {
    std::string kType, kX, kY, kW, kH, kThr, kPol, kAlpha;
    int typeNumber, fx, fy, fw, fh, pol;
    double thr, alp;

    in >> kType >> typeNumber >> kX >> fx >> kY >> fy >> kW >> fw >> kH >> fh
       >> kThr >> thr >> kPol >> pol >> kAlpha >> alp;

    bool formatOk = !in.fail() && kType == "TYPE" && kX == "X" && kY == "Y" &&
                    kW == "W" && kH == "H" && kThr == "THR" && kPol == "POL" &&
                    kAlpha == "ALPHA";
    bool valueOk = typeNumber >= 0 && typeNumber <= 4 && (pol == 1 || pol == -1) &&
                   fw > 0 && fh > 0;
    if (!formatOk || !valueOk) {
        return false;
    }

    feature = HaarFeature(static_cast<HaarType>(typeNumber), fx, fy, fw, fh);
    threshold = thr;
    polarity = pol;
    alpha = alp;
    return true;
}
