// ============================================================
// main.cpp - Chương trình nhận diện khuôn mặt Viola-Jones (PBL2)
//
// Cách dùng:
//   pbl2                              mở giao diện (tự nạp model/model.txt nếu có)
//   pbl2 gui [model.txt]              mở giao diện với model chỉ định
//   pbl2 train  <dataset> [model.txt]       huấn luyện (dataset/positive + dataset/negative)
//   pbl2 detect <anh> [model.txt] [ra.png]  phát hiện mặt, in tọa độ, lưu ảnh có khung
//   pbl2 eval   <thu_muc_test> [model.txt]  đo Accuracy / Precision / Recall
//   pbl2 synth  <thu_muc> [mat] [nen] [test]  tạo dữ liệu GIẢ LẬP để chạy thử khi chưa có dataset thật
//   pbl2 help
// (Chữ in ra console không dấu vì console Windows hay lỗi font.)
// ============================================================

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/DemoController.h"
#include "src/Evaluator.h"
#include "src/FaceDetector.h"
#include "src/GUI.h"
#include "src/Image.h"
#include "src/ProgressListener.h"
#include "src/SyntheticData.h"
#include "src/Trainer.h"
#include "src/Visualizer.h"

namespace {

const char* DEFAULT_MODEL = "model/model.txt";

// In tiến trình ra console (chỉ in khi đổi nội dung hoặc tăng đủ 10%).
class ConsoleProgress : public ProgressListener {
private:
    std::string lastMessage;
    int lastPercent;

public:
    ConsoleProgress() : lastMessage(), lastPercent(-100) {}

    void onProgress(const std::string& message, int percent) {
        // Coi là "bước mới" khi 7 ký tự đầu của nội dung đổi (ví dụ "Stage 1" -> "Stage 2").
        bool newStep = message.substr(0, 7) != lastMessage.substr(0, 7);
        if (newStep || percent >= lastPercent + 10 || percent == 100) {
            std::cout << "[" << percent << "%] " << message << std::endl;
            lastMessage = message;
            lastPercent = percent;
        }
    }
};

void printUsage() {
    std::cout << "Viola-Jones face detector (PBL2)\n\n"
              << "  pbl2                                  open the GUI (loads model/model.txt if present)\n"
              << "  pbl2 gui [model.txt]                  open the GUI with a given model\n"
              << "  pbl2 train  <dataset> [model.txt]     train from <dataset>/positive and <dataset>/negative\n"
              << "  pbl2 detect <image> [model.txt] [out.png]   detect faces, print boxes, save annotated image\n"
              << "  pbl2 eval   <test_folder> [model.txt]       Accuracy / Precision / Recall on\n"
              << "                                               <test_folder>/positive and /negative\n"
              << "  pbl2 synth  <folder> [faces=300] [backgrounds=60] [tests=40]\n"
              << "                                        create SYNTHETIC sample data (not real faces) to try the program\n"
              << "  pbl2 help\n";
}

bool loadModelOrExplain(FaceDetector& detector, const std::string& path) {
    if (!detector.loadModel(path)) {
        std::cout << "Cannot load model '" << path << "'. Train one first: pbl2 train <dataset>\n";
        return false;
    }
    return true;
}

int runGui(const std::string& modelPath) {
    GUI gui;
    std::string path = modelPath.empty() ? std::string(DEFAULT_MODEL) : modelPath;
    if (std::filesystem::exists(path)) {
        gui.loadModel(path);
    }
    gui.run();
    return 0;
}

int runTrain(const std::string& dataset, const std::string& modelPath) {
    Trainer trainer;
    TrainReport report;
    ConsoleProgress progress;
    CascadeClassifier cascade = trainer.trainFromFolders(dataset, &progress, report);

    std::vector<std::string> lines = Trainer::describe(report);
    for (size_t i = 0; i < lines.size(); i++) {
        std::cout << lines[i] << "\n";
    }
    if (!report.success) {
        return 1;
    }

    std::filesystem::path out(modelPath);
    if (out.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(out.parent_path(), error);
    }
    FaceDetector detector(cascade);
    if (!detector.saveModel(modelPath)) {
        std::cout << "Cannot write " << modelPath << "\n";
        return 1;
    }
    std::cout << "Model saved to " << modelPath << "\n";
    return 0;
}

int runDetect(const std::string& imagePath, const std::string& modelPath, const std::string& outputPath) {
    FaceDetector detector;
    if (!loadModelOrExplain(detector, modelPath)) {
        return 1;
    }
    Image image;
    if (!image.load(imagePath)) {
        std::cout << "Cannot read image '" << imagePath << "'\n";
        return 1;
    }

    DetectionStats stats;
    std::vector<Rect> faces = detector.detect(image, &stats);
    std::cout << "Image: " << image.getWidth() << " x " << image.getHeight() << "\n";
    std::cout << (faces.empty() ? "No face detected" : "Face detected") << ": " << faces.size() << " face(s)\n";
    for (size_t i = 0; i < faces.size(); i++) {
        std::cout << "  #" << i + 1 << ": x=" << faces[i].x << " y=" << faces[i].y
                  << " size " << faces[i].w << "x" << faces[i].h << "\n";
    }
    std::cout << "Pyramid levels: " << stats.levels << ", windows tested: " << stats.windowsEvaluated
              << ", raw boxes: " << stats.rawDetections << ", time: " << stats.milliseconds << " ms\n";

    Image annotated = Visualizer::toColor(image);
    for (size_t i = 0; i < faces.size(); i++) {
        Visualizer::drawOutline(annotated, faces[i], 0, 255, 0, 2);
    }
    if (annotated.save(outputPath)) {
        std::cout << "Annotated image saved to " << outputPath << "\n";
    }
    return 0;
}

int runEval(const std::string& testFolder, const std::string& modelPath) {
    FaceDetector detector;
    if (!loadModelOrExplain(detector, modelPath)) {
        return 1;
    }
    ConsoleProgress progress;
    EvaluationResult result = Evaluator::evaluateFolders(detector, testFolder, &progress);
    std::vector<std::string> lines = Evaluator::describe(result);
    for (size_t i = 0; i < lines.size(); i++) {
        std::cout << lines[i] << "\n";
    }
    if (result.totalImages() == 0) {
        return 1;
    }
    std::error_code error;
    std::filesystem::create_directories("model", error);
    if (Evaluator::saveReport(result, "model/evaluation.txt")) {
        std::cout << "Report saved to model/evaluation.txt\n";
    }
    return 0;
}

int runSynth(const std::string& folder, int faces, int backgrounds, int tests) {
    if (faces < 20 || backgrounds < 5 || tests < 1) {
        std::cout << "Need at least 20 faces, 5 backgrounds and 1 test image.\n";
        return 1;
    }
    if (!SyntheticData::writeDataset(folder, faces, backgrounds, tests)) {
        std::cout << "Cannot write the sample data into '" << folder << "'.\n";
        return 1;
    }
    std::cout << "Created SYNTHETIC sample data in '" << folder << "': " << faces << " faces, " << backgrounds
              << " backgrounds, " << tests << "+" << tests << " test images.\n"
              << "(These are not real faces: they only let you try the program.)\n\n"
              << "Next steps:\n"
              << "  pbl2 train " << folder << "\n"
              << "  pbl2 eval " << folder << "/test\n"
              << "  pbl2            (then Open Image on a file from " << folder << "/test/positive)\n";
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string command = (argc > 1) ? argv[1] : "gui";

    if (command == "gui") {
        return runGui(argc > 2 ? argv[2] : "");
    }
    if (command == "train" && argc >= 3) {
        return runTrain(argv[2], argc > 3 ? argv[3] : DEFAULT_MODEL);
    }
    if (command == "detect" && argc >= 3) {
        return runDetect(argv[2], argc > 3 ? argv[3] : DEFAULT_MODEL, argc > 4 ? argv[4] : "result.png");
    }
    if (command == "eval" && argc >= 3) {
        return runEval(argv[2], argc > 3 ? argv[3] : DEFAULT_MODEL);
    }
    if (command == "synth" && argc >= 3) {
        return runSynth(argv[2], argc > 3 ? std::atoi(argv[3]) : 300, argc > 4 ? std::atoi(argv[4]) : 60,
                        argc > 5 ? std::atoi(argv[5]) : 40);
    }
    printUsage();
    return (command == "help") ? 0 : 1;
}
