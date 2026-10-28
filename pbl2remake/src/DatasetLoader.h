#ifndef DATASETLOADER_H
#define DATASETLOADER_H

// ============================================================
// DatasetLoader.h - Đọc dataset từ thư mục
//
//   dataset/
//     positive/   ảnh khuôn mặt (nên là ảnh đã cắt sát mặt)
//     negative/   ảnh KHÔNG có khuôn mặt (phong cảnh, đồ vật, ... kích thước tùy ý)
//
// Ảnh mặt được đưa về xám, cắt vuông ở giữa, rồi thu về windowSize x windowSize.
// Ảnh nền được cắt ra NHIỀU cửa sổ windowSize x windowSize ở nhiều vị trí và tỉ lệ.
// ============================================================

#include <string>
#include <vector>
#include "Image.h"

class DatasetLoader {
public:
    // Đuôi file ảnh được chấp nhận: .jpg .jpeg .png .bmp (không phân biệt hoa thường).
    static bool isImageFile(const std::string& path);

    // Danh sách file ảnh trong thư mục, xếp theo tên. Thư mục không tồn tại -> danh sách rỗng.
    static std::vector<std::string> listImages(const std::string& folder);

    // Ảnh -> xám -> cắt vuông ở giữa -> thu về size x size.
    static Image prepareFace(const Image& image, int size);

    // Đọc ảnh mặt. Nếu có nhiều hơn maxCount ảnh thì chọn đều nhau (maxCount <= 0: lấy hết).
    // skipped (nếu có) nhận số file không đọc được.
    static std::vector<Image> loadPositives(const std::string& folder, int windowSize,
                                            int maxCount, int* skipped = 0);

    // Đọc ảnh nền (đổi sang xám). maxImages <= 0: lấy hết.
    static std::vector<Image> loadNegativeImages(const std::string& folder, int maxImages,
                                                 int* skipped = 0);

    // Cắt cửa sổ ngẫu nhiên (luân phiên từng ảnh) ở nhiều tỉ lệ, thu về windowSize x windowSize.
    // Mỗi ảnh cho tối đa windowsPerImage cửa sổ; tổng cộng tối đa maxTotal. Ảnh nhỏ hơn cửa sổ bị bỏ qua.
    static std::vector<Image> sampleNegativeWindows(const std::vector<Image>& images, int windowSize,
                                                    int windowsPerImage, int maxTotal, unsigned int seed);
};

#endif
