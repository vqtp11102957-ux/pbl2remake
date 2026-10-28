#include "DialogProvider.h"

#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#pragma comment(lib, "comdlg32.lib")
#endif

namespace {

// Nhập một dòng từ console; bỏ dấu ngoặc kép / khoảng trắng hai đầu (khi kéo thả file vào console).
std::string readLine(const std::string& prompt) {
    std::cout << prompt;
    std::string line;
    std::getline(std::cin, line);
    while (!line.empty() && (line.front() == '"' || line.front() == ' ')) {
        line.erase(line.begin());
    }
    while (!line.empty() && (line.back() == '"' || line.back() == ' ' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

#ifdef _WIN32
// Hộp thoại Open / Save của Windows. filter có dạng "Mo ta\0*.ext\0Mo ta 2\0*.*\0" (hai NUL cuối).
std::string windowsDialog(const char* title, const char* filter, bool save) {
    char fileName[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title;
    if (save) {
        ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    } else {
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    }
    BOOL ok = save ? GetSaveFileNameA(&ofn) : GetOpenFileNameA(&ofn);
    return ok ? std::string(fileName) : std::string();
}
#endif

}  // namespace

std::string DefaultDialogs::askOpenImage() {
#ifdef _WIN32
    return windowsDialog("Open image", "Images\0*.jpg;*.jpeg;*.png;*.bmp\0All files\0*.*\0", false);
#else
    return readLine("Image path (JPG/PNG, empty = cancel): ");
#endif
}

std::string DefaultDialogs::askOpenModel() {
#ifdef _WIN32
    return windowsDialog("Load model", "Model\0*.txt\0All files\0*.*\0", false);
#else
    return readLine("Model path (empty = cancel): ");
#endif
}

std::string DefaultDialogs::askSaveModel() {
#ifdef _WIN32
    return windowsDialog("Save model", "Model\0*.txt\0All files\0*.*\0", true);
#else
    return readLine("Save model as (empty = cancel): ");
#endif
}

// Thư mục: nhập trong console (cửa sổ console đi kèm chương trình).
std::string DefaultDialogs::askFolder(const std::string& prompt, const std::string& defaultPath) {
    std::string line = readLine(prompt + " [" + defaultPath + "]: ");
    return line.empty() ? defaultPath : line;
}
