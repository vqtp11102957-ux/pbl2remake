#include "FeatureGenerator.h"

// Tham số < 1 sẽ bị đưa về 1 để vòng lặp không bị treo.
FeatureGenerator::FeatureGenerator(int windowSize_, int positionStep_,
                                   int sizeStep_, int minCellSize_)
    : windowSize(windowSize_), positionStep(positionStep_),
      sizeStep(sizeStep_), minCellSize(minCellSize_) {
    if (windowSize < 1) windowSize = 1;
    if (positionStep < 1) positionStep = 1;
    if (sizeStep < 1) sizeStep = 1;
    if (minCellSize < 1) minCellSize = 1;
}

int FeatureGenerator::getWindowSize() const { return windowSize; }
int FeatureGenerator::getPositionStep() const { return positionStep; }
int FeatureGenerator::getSizeStep() const { return sizeStep; }
int FeatureGenerator::getMinCellSize() const { return minCellSize; }

// Với một loại feature (có across x down ô):
//   thử mọi kích thước ô, rồi mọi vị trí sao cho feature nằm trọn trong cửa sổ.
void FeatureGenerator::addAllOfType(HaarType type, std::vector<HaarFeature>& features) const {
    HaarFeature sample(type, 0, 0, 1, 1);      // chỉ để hỏi số ô across/down
    int across = sample.cellsAcross();
    int down = sample.cellsDown();

    for (int cellW = minCellSize; cellW * across <= windowSize; cellW += sizeStep) {
        for (int cellH = minCellSize; cellH * down <= windowSize; cellH += sizeStep) {
            int totalW = cellW * across;
            int totalH = cellH * down;

            for (int y = 0; y + totalH <= windowSize; y += positionStep) {
                for (int x = 0; x + totalW <= windowSize; x += positionStep) {
                    features.push_back(HaarFeature(type, x, y, cellW, cellH));
                }
            }
        }
    }
}

std::vector<HaarFeature> FeatureGenerator::generate() const {
    std::vector<HaarFeature> features;
    addAllOfType(TWO_HORIZONTAL, features);
    addAllOfType(TWO_VERTICAL, features);
    addAllOfType(THREE_HORIZONTAL, features);
    addAllOfType(THREE_VERTICAL, features);
    addAllOfType(FOUR_RECT, features);
    return features;
}
