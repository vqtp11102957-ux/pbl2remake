#include "TrainingSet.h"

TrainingSet::TrainingSet(int windowSize_)
    : windowSize(windowSize_), images(), sigmas(), labels() {
    if (windowSize < 1) {
        windowSize = 1;
    }
}

bool TrainingSet::add(const Image& window, Label label) {
    if (window.isEmpty() || window.getWidth() != windowSize || window.getHeight() != windowSize) {
        return false;
    }
    IntegralImage ii(window);
    images.push_back(ii);
    sigmas.push_back(ii.windowStdDev(Rect(0, 0, windowSize, windowSize)));
    labels.push_back(label);
    return true;
}

int TrainingSet::getWindowSize() const { return windowSize; }
int TrainingSet::size() const { return static_cast<int>(labels.size()); }

int TrainingSet::countFaces() const {
    int count = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == FACE) count++;
    }
    return count;
}

int TrainingSet::countNonFaces() const {
    return size() - countFaces();
}

const IntegralImage& TrainingSet::getImage(int index) const { return images[index]; }
double TrainingSet::getSigma(int index) const { return sigmas[index]; }
Label TrainingSet::getLabel(int index) const { return labels[index]; }
const std::vector<Label>& TrainingSet::getLabels() const { return labels; }
