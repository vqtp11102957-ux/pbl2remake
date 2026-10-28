#include "DatasetLoader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>

#include "Random.h"

namespace fs = std::filesystem;

bool DatasetLoader::isImageFile(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string ext = path.substr(dot);
    for (size_t i = 0; i < ext.size(); i++) {
        ext[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(ext[i])));
    }
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<std::string> DatasetLoader::listImages(const std::string& folder) {
    std::vector<std::string> files;
    std::error_code error;
    if (!fs::is_directory(folder, error)) {
        return files;
    }
    for (fs::directory_iterator it(folder, error), end; !error && it != end; it.increment(error)) {
        if (it->is_regular_file(error) && isImageFile(it->path().string())) {
            files.push_back(it->path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Cắt hình vuông lớn nhất ở giữa ảnh xám, rồi thu về size x size.
Image DatasetLoader::prepareFace(const Image& image, int size) {
    if (image.isEmpty() || size < 1) {
        return Image();
    }
    Image gray = image.toGrayscale();
    int side = std::min(gray.getWidth(), gray.getHeight());
    int x = (gray.getWidth() - side) / 2;
    int y = (gray.getHeight() - side) / 2;
    Image square = gray.crop(Rect(x, y, side, side));
    if (side == size) {
        return square;
    }
    return square.resized(size, size);
}

std::vector<Image> DatasetLoader::loadPositives(const std::string& folder, int windowSize,
                                                int maxCount, int* skipped) {
    std::vector<std::string> files = listImages(folder);
    std::vector<std::string> chosen;
    if (maxCount > 0 && static_cast<int>(files.size()) > maxCount) {
        // chọn đều nhau trong danh sách đã xếp theo tên (tránh chỉ lấy vài người đầu tiên)
        for (int i = 0; i < maxCount; i++) {
            chosen.push_back(files[static_cast<size_t>(i) * files.size() / maxCount]);
        }
    } else {
        chosen = files;
    }

    std::vector<Image> faces;
    int bad = 0;
    for (size_t i = 0; i < chosen.size(); i++) {
        Image img;
        if (img.load(chosen[i])) {
            faces.push_back(prepareFace(img, windowSize));
        } else {
            bad++;
        }
    }
    if (skipped != 0) {
        *skipped = bad;
    }
    return faces;
}

std::vector<Image> DatasetLoader::loadNegativeImages(const std::string& folder, int maxImages, int* skipped) {
    std::vector<std::string> files = listImages(folder);
    if (maxImages > 0 && static_cast<int>(files.size()) > maxImages) {
        std::vector<std::string> chosen;
        for (int i = 0; i < maxImages; i++) {
            chosen.push_back(files[static_cast<size_t>(i) * files.size() / maxImages]);
        }
        files = chosen;
    }

    std::vector<Image> images;
    int bad = 0;
    for (size_t i = 0; i < files.size(); i++) {
        Image img;
        if (img.load(files[i])) {
            images.push_back(img.toGrayscale());
        } else {
            bad++;
        }
    }
    if (skipped != 0) {
        *skipped = bad;
    }
    return images;
}

std::vector<Image> DatasetLoader::sampleNegativeWindows(const std::vector<Image>& images, int windowSize,
                                                        int windowsPerImage, int maxTotal, unsigned int seed) {
    std::vector<Image> windows;
    Random random(seed);

    for (int round = 0; round < windowsPerImage; round++) {
        bool addedAny = false;
        for (size_t i = 0; i < images.size(); i++) {
            if (maxTotal > 0 && static_cast<int>(windows.size()) >= maxTotal) {
                return windows;
            }
            const Image& img = images[i];
            int shortSide = std::min(img.getWidth(), img.getHeight());
            if (shortSide < windowSize) {
                continue;
            }

            // Chọn kích thước cửa sổ cắt: windowSize * 1.25^k, vừa trong ảnh
            int maxLevel = 0;
            while (windowSize * std::pow(1.25, maxLevel + 1) <= shortSide) {
                maxLevel++;
            }
            int level = random.nextInt(maxLevel + 1);
            int side = static_cast<int>(std::floor(windowSize * std::pow(1.25, level) + 0.5));
            if (side > shortSide) {
                side = shortSide;
            }
            int x = random.nextInt(img.getWidth() - side + 1);
            int y = random.nextInt(img.getHeight() - side + 1);

            Image piece = img.crop(Rect(x, y, side, side));
            windows.push_back(side == windowSize ? piece : piece.resized(windowSize, windowSize));
            addedAny = true;
        }
        if (!addedAny) {
            break;
        }
    }
    return windows;
}
