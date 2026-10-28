#include "GUI.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Evaluator.h"
#include "ProgressListener.h"
#include "Trainer.h"

namespace {

const char* WINDOW_NAME = "Viola-Jones Face Detector";

// Chuyển tiến trình của Trainer / Evaluator thành thanh tiến trình trên giao diện.
class GuiProgress : public ProgressListener {
private:
    GUI& gui;

public:
    explicit GuiProgress(GUI& gui_) : gui(gui_) {}

    void onProgress(const std::string& message, int percent) {
        gui.reportBusy(message, percent);
    }

    bool shouldCancel() {
        return gui.cancelWasRequested();
    }
};

// Cắt một dòng dài thành nhiều dòng ngắn (tối đa maxChars ký tự), ngắt ở khoảng trắng.
// Các dòng tiếp theo được thụt vào 2 ký tự.
std::vector<std::string> wrapText(const std::string& line, size_t maxChars) {
    std::vector<std::string> result;
    if (line.size() <= maxChars) {
        result.push_back(line);
        return result;
    }
    std::string current;
    size_t pos = 0;
    while (pos <= line.size()) {
        size_t end = line.find(' ', pos);
        if (end == std::string::npos) {
            end = line.size();
        }
        std::string word = line.substr(pos, end - pos);
        if (current.empty()) {
            current = word;
        } else if (current.size() + 1 + word.size() > maxChars) {
            result.push_back(current);
            current = "  " + word;
        } else {
            current += " " + word;
        }
        pos = end + 1;
    }
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

long long nowMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

void onMouseEvent(int event, int x, int y, int, void* userData) {
    GUI* gui = static_cast<GUI*>(userData);
    if (gui != 0) {
        gui->handleMouse(event, x, y);
    }
}

}  // namespace

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

GUI::GUI(DialogProvider* dialogs_)
    : ownDialogs(), dialogs(dialogs_ != 0 ? dialogs_ : &ownDialogs), detector(), camera(), demo(),
      original(), imageName(), gray(), grayValid(false), integralVisual(), integralVisualValid(false),
      grayIntegral(), grayIntegralValid(false), detections(), lastMs(0.0), lastWindows(0),
      mode(VIEW_ORIGINAL), running(true), buttons(), texts(), infoLines(), statusMessage(), lastFit(),
      cameraOn(false), cameraFailures(0), fps(0.0), savedParams(),
      featureType(TWO_HORIZONTAL), dragging(false), dragStartX(0), dragStartY(0), dragEndX(0), dragEndY(0),
      hasSelection(false), selection(), busy(false), cancelRequested(false), busyPercent(0),
      busyMessage(), lastBusyDraw(0) {
    layoutButtons();
    statusMessage = "Open an image or start the camera.";
}

GUI::~GUI() {
}

// ------------------------------------------------------------
// Bố cục
// ------------------------------------------------------------

void GUI::layoutButtons() {
    static const char* labels[BUTTON_COUNT] = {
        "Open Image", "Start Camera", "Stop Camera", "Train Model",
        "Load Model", "Save Model", "Evaluate", "Quit",
        "Original", "Grayscale", "Integral", "Haar Feature",
        "Detect Face", "Demo Mode", "Prev Step", "Next Step"
    };
    buttons.clear();
    for (int i = 0; i < BUTTON_COUNT; i++) {
        Button b;
        b.id = static_cast<ButtonId>(i);
        b.rect = Rect(20 + (i % 8) * 136, 48 + (i / 8) * 40, 128, 34);
        b.label = labels[i];
        buttons.push_back(b);
    }
}

Rect GUI::imageArea() const { return Rect(20, 135, 720, 520); }
Rect GUI::infoArea() const { return Rect(760, 135, 340, 520); }

// ------------------------------------------------------------
// Dữ liệu, cache
// ------------------------------------------------------------

bool GUI::hasImage() const { return !original.isEmpty(); }

void GUI::invalidateCaches() {
    grayValid = false;
    integralVisualValid = false;
    grayIntegralValid = false;
}

void GUI::ensureGray() {
    if (!grayValid) {
        gray = original.toGrayscale();
        grayValid = true;
    }
}

void GUI::ensureIntegralVisual() {
    if (!integralVisualValid) {
        ensureGrayIntegral();
        integralVisual = grayIntegral.toVisualImage();
        integralVisualValid = true;
    }
}

void GUI::ensureGrayIntegral() {
    if (!grayIntegralValid) {
        ensureGray();
        grayIntegral = IntegralImage(gray);
        grayIntegralValid = true;
    }
}

void GUI::setStatus(const std::string& message) { statusMessage = message; }

void GUI::clearSelection() {
    dragging = false;
    hasSelection = false;
    selection = Rect();
}

bool GUI::setImage(const Image& image, const std::string& name) {
    if (image.isEmpty()) {
        return false;
    }
    original = image;
    imageName = name;
    invalidateCaches();
    clearSelection();
    demo = DemoController();
    runDetection();
    mode = VIEW_DETECT;
    return true;
}

void GUI::setDetector(const FaceDetector& newDetector) {
    detector = newDetector;
    demo = DemoController();
    if (hasImage()) {
        runDetection();
    }
}

bool GUI::loadModel(const std::string& path) {
    if (!detector.loadModel(path)) {
        setStatus("Cannot load model: " + path);
        return false;
    }
    demo = DemoController();
    if (hasImage() && !cameraOn) {
        runDetection();
    }
    setStatus("Model loaded: " + path);
    return true;
}

void GUI::runDetection() {
    detections.clear();
    lastMs = 0.0;
    lastWindows = 0;
    if (!hasImage()) {
        return;
    }
    if (!detector.hasModel()) {
        setStatus("No model loaded: use Train Model or Load Model first.");
        return;
    }
    DetectionStats stats;
    detections = detector.detect(original, &stats);
    lastMs = stats.milliseconds;
    lastWindows = static_cast<int>(stats.windowsEvaluated);
    setStatus(detections.empty() ? "No face detected." : "Face detected.");
}

// ------------------------------------------------------------
// Nút
// ------------------------------------------------------------

bool GUI::isEnabled(ButtonId id) const {
    switch (id) {
    case BTN_START_CAMERA: return !cameraOn && !busy;
    case BTN_STOP_CAMERA:  return cameraOn;
    case BTN_SAVE_MODEL:   return detector.hasModel();
    case BTN_ORIGINAL:
    case BTN_GRAY:
    case BTN_INTEGRAL:
    case BTN_HAAR:         return hasImage() && !cameraOn;
    case BTN_DETECT:
    case BTN_DEMO:         return hasImage() && detector.hasModel() && !cameraOn;
    case BTN_PREV:
    case BTN_NEXT:         return mode == VIEW_DEMO && demo.isPrepared();
    default:               return true;
    }
}

bool GUI::isActive(ButtonId id) const {
    switch (id) {
    case BTN_START_CAMERA: return cameraOn;
    case BTN_ORIGINAL:     return mode == VIEW_ORIGINAL;
    case BTN_GRAY:         return mode == VIEW_GRAY;
    case BTN_INTEGRAL:     return mode == VIEW_INTEGRAL;
    case BTN_HAAR:         return mode == VIEW_HAAR;
    case BTN_DETECT:       return mode == VIEW_DETECT;
    case BTN_DEMO:         return mode == VIEW_DEMO;
    default:               return false;
    }
}

void GUI::pressButton(ButtonId id) {
    if (busy) {
        return;
    }
    if (!isEnabled(id)) {
        // Giải thích vì sao nút không dùng được
        if (id == BTN_PREV || id == BTN_NEXT) {
            setStatus("Press Demo Mode first.");
        } else if (id == BTN_SAVE_MODEL) {
            setStatus("There is no model to save.");
        } else if (id == BTN_STOP_CAMERA) {
            setStatus("The camera is not running.");
        } else if (cameraOn) {
            setStatus("Stop the camera first.");
        } else if (!hasImage()) {
            setStatus("Open an image first.");
        } else if (!detector.hasModel()) {
            setStatus("Load or train a model first.");
        }
        return;
    }

    switch (id) {
    case BTN_OPEN_IMAGE:   openImageFromDialog(); break;
    case BTN_START_CAMERA: startCamera(); break;
    case BTN_STOP_CAMERA:  stopCamera(); break;
    case BTN_TRAIN:        trainModel(); break;
    case BTN_LOAD_MODEL:   loadModelFromDialog(); break;
    case BTN_SAVE_MODEL:   saveModelFromDialog(); break;
    case BTN_EVALUATE:     evaluateModel(); break;
    case BTN_QUIT:         running = false; break;
    case BTN_ORIGINAL:
        mode = VIEW_ORIGINAL;
        setStatus("Original image.");
        break;
    case BTN_GRAY:
        ensureGray();
        mode = VIEW_GRAY;
        setStatus("Grayscale: gray = 0.299R + 0.587G + 0.114B");
        break;
    case BTN_INTEGRAL:
        ensureIntegralVisual();
        mode = VIEW_INTEGRAL;
        setStatus("Integral Image (brighter = larger running sum).");
        break;
    case BTN_HAAR:
        ensureGrayIntegral();
        mode = VIEW_HAAR;
        setStatus("Drag a rectangle on the image to place a Haar feature.");
        break;
    case BTN_DETECT:
        runDetection();
        mode = VIEW_DETECT;
        break;
    case BTN_DEMO:
        if (demo.prepare(original, detector)) {
            mode = VIEW_DEMO;
            setStatus("Demo mode: press Next Step.");
        } else {
            setStatus("Cannot start the demo.");
        }
        break;
    case BTN_PREV:
        demo.previous();
        break;
    case BTN_NEXT:
        demo.next();
        break;
    default:
        break;
    }
}

// ------------------------------------------------------------
// Hộp thoại
// ------------------------------------------------------------

void GUI::openImageFromDialog() {
    std::string path = dialogs->askOpenImage();
    if (path.empty()) {
        setStatus("Open image cancelled.");
        return;
    }
    Image image;
    if (!image.load(path)) {
        setStatus("Cannot read image: " + path);
        return;
    }
    setImage(image, std::filesystem::path(path).filename().string());
}

void GUI::loadModelFromDialog() {
    std::string path = dialogs->askOpenModel();
    if (path.empty()) {
        setStatus("Load model cancelled.");
        return;
    }
    loadModel(path);
}

void GUI::saveModelFromDialog() {
    std::string path = dialogs->askSaveModel();
    if (path.empty()) {
        setStatus("Save model cancelled.");
        return;
    }
    setStatus(detector.saveModel(path) ? "Model saved: " + path : "Cannot save model: " + path);
}

// ------------------------------------------------------------
// Webcam
// ------------------------------------------------------------

void GUI::startCamera() {
    if (cameraOn) {
        return;
    }
    if (!camera.open(0)) {
        setStatus("Cannot open the webcam (is another program using it?).");
        return;
    }
    cameraOn = true;
    cameraFailures = 0;
    fps = 0.0;
    savedParams = detector.getParams();
    DetectorParams fast = savedParams;
    fast.maxProcessingWidth = 320;               // thu nhỏ khung hình để kịp realtime
    detector.setParams(fast);
    mode = VIEW_DETECT;
    detections.clear();
    setStatus(detector.hasModel() ? "Camera started." : "Camera started (no model loaded: nothing will be detected).");
}

void GUI::stopCamera() {
    if (!cameraOn) {
        return;
    }
    camera.close();
    cameraOn = false;
    detector.setParams(savedParams);
    setStatus("Camera stopped.");
}

void GUI::update() {
    if (!cameraOn) {
        return;
    }
    long long start = nowMilliseconds();
    Image frame;
    if (!camera.read(frame)) {
        cameraFailures++;
        if (cameraFailures > 60) {
            stopCamera();
            setStatus("Camera stopped: cannot read frames.");
        }
        return;
    }
    cameraFailures = 0;
    original = frame;
    imageName = "webcam";
    invalidateCaches();
    runDetection();
    mode = VIEW_DETECT;

    double elapsed = static_cast<double>(nowMilliseconds() - start);
    double instant = (elapsed > 1.0) ? 1000.0 / elapsed : 1000.0;
    fps = (fps <= 0.0) ? instant : 0.8 * fps + 0.2 * instant;
}

// ------------------------------------------------------------
// Huấn luyện, đánh giá
// ------------------------------------------------------------

void GUI::trainModel() {
    std::string folder = dialogs->askFolder("Dataset folder (with 'positive' and 'negative')", "dataset");
    if (folder.empty()) {
        setStatus("Training cancelled.");
        return;
    }
    if (cameraOn) {
        stopCamera();
    }

    Trainer trainer;
    TrainReport report;
    GuiProgress progress(*this);
    busy = true;
    cancelRequested = false;
    busyPercent = 0;
    busyMessage = "Starting";
    CascadeClassifier cascade = trainer.trainFromFolders(folder, &progress, report);
    busy = false;

    infoLines = Trainer::describe(report);
    if (report.success) {
        detector.setCascade(cascade);
        demo = DemoController();
        std::error_code error;
        std::filesystem::create_directories("model", error);
        bool saved = detector.saveModel("model/model.txt");
        if (hasImage()) {
            runDetection();
        }
        setStatus(saved ? "Model trained and saved to model/model.txt." : "Model trained (could not save model/model.txt).");
    } else {
        setStatus(report.message.empty() ? "Training failed." : report.message);
    }
}

void GUI::evaluateModel() {
    if (!detector.hasModel()) {
        setStatus("Load or train a model first.");
        return;
    }
    std::string folder = dialogs->askFolder("Test folder (with 'positive' and 'negative')", "dataset/test");
    if (folder.empty()) {
        setStatus("Evaluation cancelled.");
        return;
    }
    if (cameraOn) {
        stopCamera();
    }

    GuiProgress progress(*this);
    busy = true;
    cancelRequested = false;
    busyPercent = 0;
    busyMessage = "Evaluating";
    EvaluationResult result = Evaluator::evaluateFolders(detector, folder, &progress);
    busy = false;

    infoLines = Evaluator::describe(result);
    if (result.totalImages() > 0) {
        std::error_code error;
        std::filesystem::create_directories("model", error);
        Evaluator::saveReport(result, "model/evaluation.txt");
        setStatus("Evaluation finished (report saved to model/evaluation.txt).");
    } else {
        setStatus("No test images found in " + folder + " (expected 'positive' and 'negative' sub-folders).");
    }
}

void GUI::reportBusy(const std::string& message, int percent) {
    busyMessage = message;
    busyPercent = std::max(0, std::min(100, percent));
    long long now = nowMilliseconds();
    if (now - lastBusyDraw >= 80) {            // vẽ lại tối đa ~12 lần / giây
        lastBusyDraw = now;
        int key = present(1);
        if (key == 27) {
            cancelRequested = true;
        }
    }
}

bool GUI::cancelWasRequested() const { return cancelRequested; }

// ------------------------------------------------------------
// Chuột và phím
// ------------------------------------------------------------

Rect GUI::activeSelection() const {
    if (dragging) {
        int x0 = std::min(dragStartX, dragEndX);
        int y0 = std::min(dragStartY, dragEndY);
        int x1 = std::max(dragStartX, dragEndX);
        int y1 = std::max(dragStartY, dragEndY);
        return Rect(x0, y0, x1 - x0, y1 - y0);
    }
    if (hasSelection) {
        return selection;
    }
    return Rect();
}

void GUI::handleMouse(int event, int x, int y) {
    Rect area = imageArea();
    bool insideArea = x >= area.x && x < area.x + area.w && y >= area.y && y < area.y + area.h;

    int imageX = 0, imageY = 0;
    if (hasImage()) {
        Visualizer::viewToImage(lastFit, x - area.x, y - area.y, imageX, imageY);
        imageX = std::max(0, std::min(original.getWidth() - 1, imageX));
        imageY = std::max(0, std::min(original.getHeight() - 1, imageY));
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (mode == VIEW_HAAR && hasImage() && !cameraOn && !busy && insideArea) {
            dragging = true;
            hasSelection = false;
            dragStartX = dragEndX = imageX;
            dragStartY = dragEndY = imageY;
        }
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (dragging) {
            dragEndX = imageX;
            dragEndY = imageY;
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        if (dragging) {
            dragEndX = imageX;
            dragEndY = imageY;
            Rect region = activeSelection();
            dragging = false;
            if (region.w >= 4 && region.h >= 4) {
                selection = region;
                hasSelection = true;
                setStatus("Feature placed. Press keys 1-5 to try other types.");
            } else {
                setStatus("Selection too small: drag a bigger rectangle.");
            }
            return;
        }
        if (busy) {
            return;
        }
        for (size_t i = 0; i < buttons.size(); i++) {
            const Rect& r = buttons[i].rect;
            if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) {
                pressButton(buttons[i].id);
                break;
            }
        }
    }
}

void GUI::handleKey(int key) {
    if (key < 0) {
        return;
    }
    int k = key & 0xFF;
    if (k == 27) {                                 // ESC: hủy việc đang chạy
        if (busy) {
            cancelRequested = true;
        }
        return;
    }
    if (busy) {
        return;
    }
    if (k >= '1' && k <= '5') {
        featureType = static_cast<HaarType>(k - '1');
        return;
    }
    switch (k) {
    case 'q': case 'Q': running = false; break;
    case 'n': case 'N': pressButton(BTN_NEXT); break;
    case 'p': case 'P': pressButton(BTN_PREV); break;
    case 'd': case 'D': pressButton(BTN_DETECT); break;
    case 'o': case 'O': pressButton(BTN_OPEN_IMAGE); break;
    default: break;
    }
}

// ------------------------------------------------------------
// Vẽ
// ------------------------------------------------------------

void GUI::addText(int x, int y, const std::string& text, int red, int green, int blue, double scale) {
    TextItem item;
    item.x = x;
    item.y = y;
    item.text = text;
    item.red = red;
    item.green = green;
    item.blue = blue;
    item.scale = scale;
    texts.push_back(item);
}

void GUI::drawButtons(Image& canvas) {
    for (size_t i = 0; i < buttons.size(); i++) {
        const Button& b = buttons[i];
        bool enabled = isEnabled(b.id);
        bool active = isActive(b.id);
        int red = 70, green = 72, blue = 90;
        if (!enabled) {
            red = 52; green = 52; blue = 58;
        } else if (active) {
            red = 30; green = 120; blue = 200;
        }
        canvas.fillRect(b.rect, static_cast<unsigned char>(red), static_cast<unsigned char>(green), static_cast<unsigned char>(blue));
        Visualizer::drawOutline(canvas, b.rect, 110, 110, 130, 1);
        int textWidth = static_cast<int>(b.label.size()) * 9;
        int textColor = enabled ? 255 : 120;
        addText(b.rect.x + (b.rect.w - textWidth) / 2, b.rect.y + 23, b.label, textColor, textColor, textColor, 0.5);
    }
}

Image GUI::buildMainView(const Rect& area) {
    lastFit = FitInfo();
    if (!hasImage()) {
        Image empty(area.w, area.h, 3, 30);
        if (!busy) {
            addText(area.x + 170, area.y + area.h / 2, "Open an image or start the camera", 180, 180, 190, 0.6);
        }
        return empty;
    }

    switch (mode) {
    case VIEW_ORIGINAL:
        return Visualizer::fitInto(original, area.w, area.h, &lastFit);
    case VIEW_GRAY:
        ensureGray();
        return Visualizer::fitInto(gray, area.w, area.h, &lastFit);
    case VIEW_INTEGRAL:
        ensureIntegralVisual();
        return Visualizer::fitInto(integralVisual, area.w, area.h, &lastFit);
    case VIEW_HAAR: {
        ensureGray();
        ensureGrayIntegral();
        Rect region = activeSelection();
        if (region.w >= 2 && region.h >= 2) {
            HaarFeature feature = HaarFeature::fromRegion(featureType, region);
            return Visualizer::renderFeature(gray, feature, area.w, area.h, &lastFit);
        }
        return Visualizer::fitInto(gray, area.w, area.h, &lastFit);
    }
    case VIEW_DETECT: {
        Image view = Visualizer::fitInto(original, area.w, area.h, &lastFit);
        for (size_t i = 0; i < detections.size(); i++) {
            Rect box = Visualizer::mapRect(detections[i], lastFit);
            Visualizer::drawOutline(view, box, 0, 255, 0, 2);
            if (!busy) {                         // đang huấn luyện thì không vẽ chữ đè lên thanh tiến trình
                addText(area.x + box.x, area.y + std::max(14, box.y - 6), "Face detected", 0, 255, 0, 0.5);
            }
        }
        return view;
    }
    case VIEW_DEMO:
        return demo.render(area.w, area.h);
    }
    return Image(area.w, area.h, 3, 30);
}

void GUI::drawBusyOverlay(Image& canvas, const Rect& area) {
    Rect box(area.x + 60, area.y + area.h / 2 - 45, area.w - 120, 90);
    canvas.fillRect(box, 18, 18, 24);
    Visualizer::drawOutline(canvas, box, 120, 120, 150, 2);
    Rect track(box.x + 20, box.y + 48, box.w - 40, 16);
    canvas.fillRect(track, 70, 70, 84);
    canvas.fillRect(Rect(track.x, track.y, track.w * busyPercent / 100, track.h), 60, 180, 90);

    addText(box.x + 20, box.y + 28, busyMessage, 255, 255, 255, 0.55);
    addText(box.x + box.w - 60, box.y + 28, std::to_string(busyPercent) + "%", 255, 255, 255, 0.55);
    addText(box.x + 20, box.y + 82, "Press ESC to cancel", 170, 170, 180, 0.45);
}

std::vector<std::string> GUI::haarInfoLines() {
    std::vector<std::string> lines;
    lines.push_back("SHOW HAAR FEATURE");
    lines.push_back("Drag a rectangle on the image.");
    lines.push_back("Keys 1-5 choose the feature type:");
    static const char* names[5] = {"2 rectangles, horizontal", "2 rectangles, vertical",
                                   "3 rectangles, horizontal", "3 rectangles, vertical", "4 rectangles"};
    for (int i = 0; i < 5; i++) {
        lines.push_back(std::string(i == static_cast<int>(featureType) ? " > [" : "   [") + std::to_string(i + 1) + "] " + names[i]);
    }
    lines.push_back("");

    Rect region = activeSelection();
    if (region.w >= 2 && region.h >= 2) {
        ensureGrayIntegral();
        HaarFeature feature = HaarFeature::fromRegion(featureType, region);
        long long whiteSum = 0, blackSum = 0;
        std::vector<Rect> whites = feature.getWhiteRects();
        std::vector<Rect> blacks = feature.getBlackRects();
        for (size_t i = 0; i < whites.size(); i++) whiteSum += grayIntegral.sumRect(whites[i]);
        for (size_t i = 0; i < blacks.size(); i++) blackSum += grayIntegral.sumRect(blacks[i]);

        lines.push_back("Selection: x=" + std::to_string(region.x) + " y=" + std::to_string(region.y) +
                        " " + std::to_string(region.w) + "x" + std::to_string(region.h));
        lines.push_back("Feature: " + feature.getTypeName());
        lines.push_back("Cell size: " + std::to_string(feature.getW()) + "x" + std::to_string(feature.getH()));
        lines.push_back("White sum (green) = " + std::to_string(whiteSum));
        lines.push_back("Black sum (red)   = " + std::to_string(blackSum));
        lines.push_back("Value = white - " + std::to_string(feature.getBlackWeight()) + " * black");
        lines.push_back("      = " + std::to_string(feature.compute(grayIntegral)));
    } else {
        lines.push_back("No feature placed yet.");
    }
    return lines;
}

std::vector<std::string> GUI::currentInfoLines() {
    std::vector<std::string> lines;
    if (mode == VIEW_DEMO && demo.isPrepared()) {
        lines.push_back(demo.getTitle());
        lines.push_back("");
        std::vector<std::string> demoLines = demo.getLines();
        lines.insert(lines.end(), demoLines.begin(), demoLines.end());
        return lines;
    }
    if (mode == VIEW_HAAR && hasImage()) {
        return haarInfoLines();
    }

    char buffer[120];
    if (hasImage()) {
        if (cameraOn) {
            lines.push_back("WEBCAM DETECTION");
            std::snprintf(buffer, sizeof(buffer), "Frame: %d x %d", original.getWidth(), original.getHeight());
            lines.push_back(buffer);
            std::snprintf(buffer, sizeof(buffer), "Speed: %.1f FPS", fps);
            lines.push_back(buffer);
        } else {
            switch (mode) {
            case VIEW_ORIGINAL:
                lines.push_back("ORIGINAL IMAGE");
                lines.push_back(imageName);
                std::snprintf(buffer, sizeof(buffer), "Size: %d x %d", original.getWidth(), original.getHeight());
                lines.push_back(buffer);
                break;
            case VIEW_GRAY:
                lines.push_back("GRAYSCALE");
                lines.push_back("gray = 0.299*R + 0.587*G + 0.114*B");
                lines.push_back("Haar features only need light/dark,");
                lines.push_back("so color is dropped.");
                break;
            case VIEW_INTEGRAL:
                lines.push_back("INTEGRAL IMAGE");
                lines.push_back("S(x,y) = sum of all pixels above-left");
                lines.push_back("of (x,y). Drawn brighter = larger sum.");
                lines.push_back("Any rectangle sum = D - B - C + A");
                lines.push_back("(4 lookups, whatever the size).");
                break;
            default:
                lines.push_back("DETECTION RESULT");
                std::snprintf(buffer, sizeof(buffer), "Windows tested: %d", lastWindows);
                lines.push_back(buffer);
                break;
            }
        }
        lines.push_back("");
        if (!detector.hasModel()) {
            lines.push_back("No model loaded.");
        } else {
            std::snprintf(buffer, sizeof(buffer), "Faces detected: %d", static_cast<int>(detections.size()));
            lines.push_back(buffer);
            for (size_t i = 0; i < detections.size() && i < 8; i++) {
                std::snprintf(buffer, sizeof(buffer), "  #%d: x=%d y=%d size %dx%d", static_cast<int>(i) + 1,
                              detections[i].x, detections[i].y, detections[i].w, detections[i].h);
                lines.push_back(buffer);
            }
        }
    } else {
        lines.push_back("No image yet.");
        lines.push_back("Use Open Image or Start Camera.");
    }

    if (!infoLines.empty()) {
        lines.push_back("");
        lines.push_back("---- last report ----");
        lines.insert(lines.end(), infoLines.begin(), infoLines.end());
    }
    return lines;
}

void GUI::drawInfoPanel(Image& canvas) {
    Rect area = infoArea();
    canvas.fillRect(area, 28, 28, 34);
    Visualizer::drawOutline(canvas, area, 80, 80, 100, 1);

    std::vector<std::string> lines = currentInfoLines();
    int y = area.y + 26;
    for (size_t i = 0; i < lines.size(); i++) {
        std::vector<std::string> wrapped = wrapText(lines[i], 40);
        for (size_t w = 0; w < wrapped.size(); w++) {
            if (y > area.y + area.h - 8) {
                return;
            }
            bool heading = (i == 0);
            int red = heading ? 255 : 220, green = heading ? 220 : 220, blue = heading ? 80 : 225;
            addText(area.x + 12, y, wrapped[w], red, green, blue, 0.45);
            y += 21;
        }
    }
}

void GUI::drawStatusBar(Image& canvas) {
    canvas.fillRect(Rect(0, 665, CANVAS_WIDTH, CANVAS_HEIGHT - 665), 20, 20, 26);

    char buffer[160];
    if (detector.hasModel()) {
        std::snprintf(buffer, sizeof(buffer), "Model: Loaded (%d stages, %d weak classifiers)",
                      detector.getCascade().stageCount(), detector.getCascade().totalWeakCount());
        addText(20, 688, buffer, 120, 230, 120, 0.5);
    } else {
        addText(20, 688, "Model: Not loaded", 240, 150, 100, 0.5);
    }
    std::snprintf(buffer, sizeof(buffer), "Faces detected: %d", static_cast<int>(detections.size()));
    addText(440, 688, buffer, 230, 230, 235, 0.5);
    std::snprintf(buffer, sizeof(buffer), "Processing time: %.0f ms", lastMs);
    addText(640, 688, buffer, 230, 230, 235, 0.5);
    if (cameraOn) {
        std::snprintf(buffer, sizeof(buffer), "%.1f FPS", fps);
        addText(950, 688, buffer, 230, 230, 235, 0.5);
    }
    addText(20, 710, statusMessage, 240, 210, 120, 0.5);
}

Image GUI::render() {
    texts.clear();
    Image canvas(CANVAS_WIDTH, CANVAS_HEIGHT, 3);
    canvas.fillRect(Rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), 38, 38, 44);
    canvas.fillRect(Rect(0, 0, CANVAS_WIDTH, 40), 20, 60, 110);
    addText(20, 28, "VIOLA-JONES FACE DETECTOR", 255, 255, 255, 0.8);

    drawButtons(canvas);

    Rect area = imageArea();
    Image view = buildMainView(area);
    canvas.paste(view, area.x, area.y);
    Visualizer::drawOutline(canvas, Rect(area.x - 1, area.y - 1, area.w + 2, area.h + 2), 80, 80, 100, 1);
    if (busy) {
        drawBusyOverlay(canvas, area);
    }

    drawInfoPanel(canvas);
    drawStatusBar(canvas);
    return canvas;
}

int GUI::present(int waitMilliseconds) {
    Image canvas = render();
    cv::Mat mat = canvas.toMat();
    for (size_t i = 0; i < texts.size(); i++) {
        const TextItem& t = texts[i];
        cv::putText(mat, t.text, cv::Point(t.x, t.y), cv::FONT_HERSHEY_SIMPLEX, t.scale,
                    cv::Scalar(t.blue, t.green, t.red), 1);
    }
    cv::imshow(WINDOW_NAME, mat);
    return cv::waitKey(waitMilliseconds);
}

void GUI::run() {
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(WINDOW_NAME, onMouseEvent, this);
    running = true;
    while (running) {
        update();
        int key = present(15);
        handleKey(key);
        if (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) < 1) {
            break;                                // người dùng đóng cửa sổ
        }
    }
    stopCamera();
    cv::destroyAllWindows();
}

// ------------------------------------------------------------
// Truy vấn
// ------------------------------------------------------------

Rect GUI::getSelection() const { return hasSelection ? selection : Rect(); }
GUI::ViewMode GUI::getMode() const { return mode; }
bool GUI::isCameraOn() const { return cameraOn; }
bool GUI::isBusy() const { return busy; }
const std::vector<Rect>& GUI::getDetections() const { return detections; }
const std::vector<GUI::Button>& GUI::getButtons() const { return buttons; }
const std::vector<GUI::TextItem>& GUI::getTexts() const { return texts; }
const std::string& GUI::getStatusMessage() const { return statusMessage; }
const FaceDetector& GUI::getDetector() const { return detector; }
const DemoController& GUI::getDemo() const { return demo; }
bool GUI::wantsQuit() const { return !running; }
bool GUI::hasFeatureSelection() const { return hasSelection; }
HaarType GUI::getFeatureType() const { return featureType; }
const std::vector<std::string>& GUI::getInfoLines() const { return infoLines; }
