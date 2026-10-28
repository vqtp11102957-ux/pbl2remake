#ifndef GUI_H
#define GUI_H

// ============================================================
// GUI.h - Giao diện chương trình (OpenCV HighGUI)
//
// Toàn bộ giao diện là MỘT cửa sổ OpenCV. Mọi nút, ô, khung đều do ta tự vẽ lên
// một ảnh (canvas) rồi hiển thị; chuột và phím được xử lý bằng hàm của chính class này.
//
//   ┌────────────────────────────────────────────────────────────────┐
//   │ VIOLA-JONES FACE DETECTOR                                      │
//   │ [Open Image][Start Camera][Stop Camera][Train][Load][Save][Eval][Quit]
//   │ [Original][Grayscale][Integral][Haar Feature][Detect][Demo][Prev][Next]
//   ├───────────────────────────────────────┬────────────────────────┤
//   │                                       │  giải thích / số liệu  │
//   │          ẢNH / WEBCAM                 │  của chế độ hiện tại   │
//   ├───────────────────────────────────────┴────────────────────────┤
//   │ Model: Loaded | Faces detected: 2 | Processing time: 35 ms     │
//   └────────────────────────────────────────────────────────────────┘
//
// Để kiểm thử được mà không cần mở cửa sổ, GUI tách rõ:
//   handleMouse / handleKey / update : thay đổi trạng thái
//   render()                          : vẽ trạng thái hiện tại ra một Image
//   present()                         : đưa Image đó lên cửa sổ (cv::imshow)
// ============================================================

#include <string>
#include <vector>
#include "CameraSource.h"
#include "DemoController.h"
#include "DialogProvider.h"
#include "FaceDetector.h"
#include "HaarFeature.h"
#include "Image.h"
#include "IntegralImage.h"
#include "Visualizer.h"

class GUI {
public:
    enum ViewMode { VIEW_ORIGINAL, VIEW_GRAY, VIEW_INTEGRAL, VIEW_HAAR, VIEW_DETECT, VIEW_DEMO };

    enum ButtonId {
        BTN_OPEN_IMAGE, BTN_START_CAMERA, BTN_STOP_CAMERA, BTN_TRAIN,
        BTN_LOAD_MODEL, BTN_SAVE_MODEL, BTN_EVALUATE, BTN_QUIT,
        BTN_ORIGINAL, BTN_GRAY, BTN_INTEGRAL, BTN_HAAR,
        BTN_DETECT, BTN_DEMO, BTN_PREV, BTN_NEXT,
        BUTTON_COUNT
    };

    struct Button {
        ButtonId id;
        Rect rect;
        std::string label;
    };

    struct TextItem {                    // một dòng chữ sẽ được vẽ bằng cv::putText
        int x;
        int y;
        std::string text;
        int red;
        int green;
        int blue;
        double scale;
    };

    static constexpr int CANVAS_WIDTH = 1120;
    static constexpr int CANVAS_HEIGHT = 720;

private:
    // ---- Dịch vụ ----
    DefaultDialogs ownDialogs;
    DialogProvider* dialogs;
    FaceDetector detector;
    CameraSource camera;
    DemoController demo;

    // ---- Dữ liệu hiển thị ----
    Image original;                      // ảnh đang xem (hoặc khung hình webcam mới nhất)
    std::string imageName;
    Image gray;                          // cache: ảnh xám của `original`
    bool grayValid;
    Image integralVisual;                // cache: Integral Image để nhìn
    bool integralVisualValid;
    IntegralImage grayIntegral;          // cache: dùng cho Haar Feature
    bool grayIntegralValid;
    std::vector<Rect> detections;
    double lastMs;
    int lastWindows;

    // ---- Trạng thái giao diện ----
    ViewMode mode;
    bool running;
    std::vector<Button> buttons;
    std::vector<TextItem> texts;
    std::vector<std::string> infoLines;  // báo cáo gần nhất (huấn luyện / đánh giá), hiện ở panel bên phải
    std::string statusMessage;
    FitInfo lastFit;                     // cách ảnh được đặt vào khung hiển thị lần vẽ gần nhất

    // ---- Webcam ----
    bool cameraOn;
    int cameraFailures;
    double fps;
    DetectorParams savedParams;

    // ---- Haar feature ----
    HaarType featureType;
    bool dragging;
    int dragStartX, dragStartY, dragEndX, dragEndY;   // tọa độ ảnh
    bool hasSelection;
    Rect selection;                      // tọa độ ảnh

    // ---- Huấn luyện / đánh giá ----
    bool busy;
    bool cancelRequested;
    int busyPercent;
    std::string busyMessage;
    long long lastBusyDraw;

    // Không cho sao chép (có webcam bên trong).
    GUI(const GUI&);
    GUI& operator=(const GUI&);

    // ---- Hàm hỗ trợ ----
    void layoutButtons();
    void invalidateCaches();
    void ensureGray();
    void ensureIntegralVisual();
    void ensureGrayIntegral();
    void runDetection();
    void setStatus(const std::string& message);
    void clearSelection();
    Rect activeSelection() const;        // vùng đang kéo / đã chọn (w = 0 nếu chưa có)
    bool isEnabled(ButtonId id) const;
    bool isActive(ButtonId id) const;
    std::vector<std::string> haarInfoLines();
    std::vector<std::string> currentInfoLines();
    void addText(int x, int y, const std::string& text, int red, int green, int blue, double scale);
    void drawButtons(Image& canvas);
    Image buildMainView(const Rect& area);
    void drawBusyOverlay(Image& canvas, const Rect& area);
    void drawInfoPanel(Image& canvas);
    void drawStatusBar(Image& canvas);
    void startCamera();
    void stopCamera();
    void trainModel();
    void evaluateModel();
    void loadModelFromDialog();
    void saveModelFromDialog();
    void openImageFromDialog();

public:
    // dialogs = nullptr: dùng DefaultDialogs.
    explicit GUI(DialogProvider* dialogs = 0);
    ~GUI();

    // ---- Dữ liệu từ bên ngoài ----
    bool setImage(const Image& image, const std::string& name);   // hiển thị ảnh và chạy phát hiện
    bool loadModel(const std::string& path);
    void setDetector(const FaceDetector& newDetector);

    // ---- Tương tác ----
    // event: các hằng cv::EVENT_*. Tọa độ (x, y) tính trên canvas.
    void handleMouse(int event, int x, int y);
    void handleKey(int key);
    void pressButton(ButtonId id);        // giống hệt bấm chuột lên nút
    void update();                        // xử lý webcam (gọi mỗi vòng lặp)

    // ---- Vẽ ----
    Image render();                       // vẽ giao diện ra ảnh CANVAS_WIDTH x CANVAS_HEIGHT; ghi sẵn danh sách chữ
    int present(int waitMilliseconds);    // render + hiển thị + chờ phím; trả về mã phím (-1/0 nếu không có)
    void run();                           // vòng lặp chính

    // ---- Dùng bởi lớp báo tiến trình ----
    void reportBusy(const std::string& message, int percent);
    bool cancelWasRequested() const;

    // ---- Truy vấn (cho kiểm thử) ----
    Rect imageArea() const;               // vùng hiển thị ảnh trên canvas
    Rect infoArea() const;                // panel thông tin bên phải
    Rect getSelection() const;            // vùng Haar feature đang chọn (w = 0 nếu chưa có)
    ViewMode getMode() const;
    bool isCameraOn() const;
    bool isBusy() const;
    const std::vector<Rect>& getDetections() const;
    const std::vector<Button>& getButtons() const;
    const std::vector<TextItem>& getTexts() const;
    const std::string& getStatusMessage() const;
    const FaceDetector& getDetector() const;
    const DemoController& getDemo() const;
    bool hasImage() const;
    bool wantsQuit() const;
    bool hasFeatureSelection() const;
    HaarType getFeatureType() const;
    const std::vector<std::string>& getInfoLines() const;
};

#endif
