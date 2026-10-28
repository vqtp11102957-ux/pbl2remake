#ifndef TRAININGSET_H
#define TRAININGSET_H

// ============================================================
// TrainingSet.h - Tập mẫu huấn luyện
//
// Mỗi mẫu là một cửa sổ ảnh (ví dụ 24x24) kèm nhãn FACE / NON_FACE.
// Với mỗi mẫu ta lưu sẵn:
//   - IntegralImage của cửa sổ (để tính Haar feature nhanh)
//   - sigma = độ lệch chuẩn của cửa sổ (để chuẩn hóa độ sáng)
//   - nhãn
// Ba danh sách này luôn có cùng độ dài; mẫu thứ i nằm ở vị trí i của cả ba.
// ============================================================

#include <vector>
#include "Image.h"
#include "IntegralImage.h"
#include "Types.h"

class TrainingSet {
private:
    int windowSize;
    std::vector<IntegralImage> images;
    std::vector<double> sigmas;
    std::vector<Label> labels;

public:
    explicit TrainingSet(int windowSize = 24);

    // Thêm một mẫu. Cửa sổ phải đúng kích thước windowSize x windowSize
    // (ảnh màu sẽ được tự đổi sang xám). Trả về false nếu không hợp lệ.
    bool add(const Image& window, Label label);

    int getWindowSize() const;
    int size() const;
    int countFaces() const;
    int countNonFaces() const;

    // index phải nằm trong [0, size()).
    const IntegralImage& getImage(int index) const;
    double getSigma(int index) const;
    Label getLabel(int index) const;
    const std::vector<Label>& getLabels() const;
};

#endif
