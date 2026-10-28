#ifndef FEATUREGENERATOR_H
#define FEATUREGENERATOR_H

// ============================================================
// FeatureGenerator.h - Sinh danh sách Haar feature cho một cửa sổ
//
// Với mỗi loại feature, ta duyệt:
//   - kích thước một ô (cellW x cellH): từ minCellSize, mỗi lần tăng sizeStep
//   - vị trí (x, y) trong cửa sổ:        từ 0, mỗi lần tăng positionStep
// và chỉ giữ các feature nằm trọn trong cửa sổ.
//
// Với cửa sổ 24x24:
//   positionStep=1, sizeStep=1, minCellSize=1  -> 162.336 feature (đầy đủ)
//   positionStep=2, sizeStep=2, minCellSize=4  ->   5.509 feature (mặc định PBL2)
//   positionStep=3, sizeStep=2, minCellSize=4  ->   2.577 feature (nhẹ)
//   positionStep=2, sizeStep=2, minCellSize=2  ->  10.344 feature (chi tiết hơn)
// ============================================================

#include <vector>
#include "HaarFeature.h"

class FeatureGenerator {
private:
    int windowSize;
    int positionStep;
    int sizeStep;
    int minCellSize;

    // Thêm vào "features" mọi feature của MỘT loại.
    void addAllOfType(HaarType type, std::vector<HaarFeature>& features) const;

public:
    FeatureGenerator(int windowSize = 24, int positionStep = 2,
                     int sizeStep = 2, int minCellSize = 4);

    int getWindowSize() const;
    int getPositionStep() const;
    int getSizeStep() const;
    int getMinCellSize() const;

    // Sinh toàn bộ feature (5 loại, nhóm theo loại).
    std::vector<HaarFeature> generate() const;
};

#endif
