#ifndef PROGRESSLISTENER_H
#define PROGRESSLISTENER_H

// ============================================================
// ProgressListener.h - "Người nghe tiến trình" cho các tác vụ lâu
//
// Đây là lớp TRỪU TƯỢNG (interface): Trainer / AdaBoost / Evaluator chỉ biết
// gọi onProgress(...) mà không cần biết ai nhận. Nhờ đó:
//   - giao diện (GUI) kế thừa để vẽ thanh tiến trình;
//   - chương trình dòng lệnh kế thừa để in ra màn hình;
//   - bài test kế thừa để ghi lại các lần được gọi.
// ============================================================

#include <string>

class ProgressListener {
public:
    virtual ~ProgressListener() {}

    // message: mô tả việc đang làm; percent: 0..100
    virtual void onProgress(const std::string& message, int percent) = 0;

    // Tác vụ kiểm tra hàm này định kỳ; trả về true nghĩa là người dùng muốn HỦY.
    virtual bool shouldCancel() { return false; }
};

#endif
