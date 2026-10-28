#ifndef DIALOGPROVIDER_H
#define DIALOGPROVIDER_H

// ============================================================
// DialogProvider.h - Cách GUI hỏi người dùng chọn file / thư mục
//
// Là lớp TRỪU TƯỢNG: GUI chỉ biết gọi askOpenImage(), ... mà không cần biết
// hộp thoại được tạo ra thế nào.
//   - DefaultDialogs: hộp thoại thật của Windows (còn các hệ khác: nhập đường dẫn trong console)
//   - bài test: một lớp giả trả về đường dẫn có sẵn, không cần ngồi bấm chuột
// Mọi hàm trả về chuỗi rỗng nếu người dùng bấm Cancel.
// ============================================================

#include <string>

class DialogProvider {
public:
    virtual ~DialogProvider() {}
    virtual std::string askOpenImage() = 0;
    virtual std::string askOpenModel() = 0;
    virtual std::string askSaveModel() = 0;
    virtual std::string askFolder(const std::string& prompt, const std::string& defaultPath) = 0;
};

// Cài đặt mặc định (xem FileDialog.cpp).
class DefaultDialogs : public DialogProvider {
public:
    std::string askOpenImage();
    std::string askOpenModel();
    std::string askSaveModel();
    std::string askFolder(const std::string& prompt, const std::string& defaultPath);
};

#endif
