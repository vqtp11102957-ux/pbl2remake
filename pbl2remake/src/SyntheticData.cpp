#include "SyntheticData.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace {

// Vẽ một dải tối (hình chữ nhật) lên ảnh xám.
void drawBand(Image& w, int x0, int y0, int bw, int bh, int value) {
    w.fillRect(Rect(x0, y0, bw, bh), static_cast<unsigned char>(value), 0, 0);
}

// Cộng nhiễu ngẫu nhiên +-20 vào mọi pixel.
void addNoise(Image& w, Random& random) {
    for (int y = 0; y < w.getHeight(); y++) {
        for (int x = 0; x < w.getWidth(); x++) {
            int v = w.getPixel(x, y) + random.nextInt(41) - 20;
            w.setPixel(x, y, 0, static_cast<unsigned char>(std::max(0, std::min(255, v))));
        }
    }
}

bool writeOne(const std::filesystem::path& folder, const char* pattern, int index, const Image& image) {
    char name[64];
    std::snprintf(name, sizeof(name), pattern, index);
    return image.save((folder / name).string());
}

}  // namespace

// "Mặt": luôn có dải tối "mắt" (hàng 6-7 đến 9-10) VÀ dải tối "miệng" (hàng 16-17).
Image SyntheticData::makeFace(Random& random) {
    Image w(24, 24, 1);
    w.fillRect(Rect(0, 0, 24, 24), static_cast<unsigned char>(150 + random.nextInt(61)), 0, 0);
    int dark = 40 + random.nextInt(50);
    int eyeTop = 6 + random.nextInt(2);
    int eyeHeight = 3 + random.nextInt(2);
    int mouthTop = 16 + random.nextInt(2);
    drawBand(w, 3, eyeTop, 18, eyeHeight, dark);
    drawBand(w, 8, mouthTop, 8, 2, dark);
    addNoise(w, random);
    return w;
}

// "Nền": 1-3 dải tối ở vị trí và kích thước ngẫu nhiên (ngang hoặc dọc).
Image SyntheticData::makeBackground(Random& random) {
    Image w(24, 24, 1);
    w.fillRect(Rect(0, 0, 24, 24), static_cast<unsigned char>(110 + random.nextInt(111)), 0, 0);
    int bands = 1 + random.nextInt(3);
    for (int b = 0; b < bands; b++) {
        int dark = 30 + random.nextInt(120);
        bool horizontal = (random.nextInt(2) == 0);
        int a = random.nextInt(16);
        int c = random.nextInt(22);
        int length = 4 + random.nextInt(14);
        int thickness = 1 + random.nextInt(5);
        if (horizontal) {
            drawBand(w, a, c, length, thickness, dark);
        } else {
            drawBand(w, c, a, thickness, length, dark);
        }
    }
    addNoise(w, random);
    return w;
}

Image SyntheticData::makeBackgroundImage(Random& random, int size) {
    Image img(size, size, 1);
    for (int y = 0; y < size; y += 24) {
        for (int x = 0; x < size; x += 24) {
            img.paste(makeBackground(random), x, y);
        }
    }
    return img;
}

bool SyntheticData::writeDataset(const std::string& root, int faces, int backgrounds, int tests) {
    std::filesystem::path base(root);
    std::error_code error;
    std::filesystem::create_directories(base / "positive", error);
    std::filesystem::create_directories(base / "negative", error);
    std::filesystem::create_directories(base / "test" / "positive", error);
    std::filesystem::create_directories(base / "test" / "negative", error);
    if (error) {
        return false;
    }

    Random random(2025);
    bool ok = true;
    for (int i = 0; i < faces; i++) {
        Image face = makeFace(random).resized(48, 48);
        ok = writeOne(base / "positive", "face_%04d.png", i, face) && ok;
    }
    for (int i = 0; i < backgrounds; i++) {
        Image background = makeBackgroundImage(random, 192);
        ok = writeOne(base / "negative", "background_%04d.png", i, background) && ok;
    }
    for (int i = 0; i < tests; i++) {
        // Ảnh thử có mặt: ảnh nền + MỘT khuôn mặt dán ở vị trí ngẫu nhiên.
        // (mỗi lệnh gọi hàm ngẫu nhiên viết riêng một dòng: thứ tự tính tham số của C++ là không xác định)
        Image scene = makeBackgroundImage(random, 160);
        int size = 48 + 8 * random.nextInt(4);
        Image face = makeFace(random).resized(size, size);
        int x = random.nextInt(160 - size + 1);
        int y = random.nextInt(160 - size + 1);
        scene.paste(face, x, y);
        ok = writeOne(base / "test" / "positive", "test_face_%04d.png", i, scene) && ok;

        Image negative = makeBackgroundImage(random, 160);
        ok = writeOne(base / "test" / "negative", "test_background_%04d.png", i, negative) && ok;
    }
    return ok;
}
