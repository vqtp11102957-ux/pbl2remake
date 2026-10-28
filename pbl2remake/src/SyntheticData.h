#ifndef SYNTHETICDATA_H
#define SYNTHETICDATA_H

// ============================================================
// SyntheticData.h - Tạo bộ dữ liệu GIẢ LẬP để chạy thử cả chương trình
//
//   "Mặt" giả lập: nền sáng có một dải tối ngang (như hai mắt) và một dải tối ngắn bên dưới
//   (như miệng). "Nền": vài dải tối ngẫu nhiên. Đây KHÔNG phải ảnh khuôn mặt thật;
//   mục đích là kiểm tra chương trình chạy thông suốt khi bạn chưa có dataset thật.
//
//   writeDataset(thu_muc, ...) tạo:
//     thu_muc/positive          ảnh "mặt" 48x48 (huấn luyện)
//     thu_muc/negative          ảnh "nền" 192x192 (huấn luyện)
//     thu_muc/test/positive     ảnh nền 160x160 có dán MỘT "mặt" 48..72 pixel (kiểm tra)
//     thu_muc/test/negative     ảnh "nền" 160x160 (kiểm tra)
//   Cùng tham số thì cho cùng dữ liệu trên mọi máy (dùng bộ ngẫu nhiên Random có seed cố định).
// ============================================================

#include <string>
#include "Image.h"
#include "Random.h"

class SyntheticData {
public:
    static Image makeFace(Random& random);                           // cửa sổ "mặt" 24x24
    static Image makeBackground(Random& random);                     // cửa sổ "nền" 24x24
    static Image makeBackgroundImage(Random& random, int size);      // ảnh nền ghép từ các ô 24x24

    // Ghi bộ dữ liệu. Trả về false nếu không ghi được một file nào đó.
    static bool writeDataset(const std::string& root, int faces, int backgrounds, int tests);
};

#endif
