#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "PlateLocator.hpp"
#include "image_utils.hpp"
#include "dataset_utils.hpp"
#include "model.hpp"
#include "recognize_utils.hpp"

std::string getCurrentTimestamp() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

int main(int argc, char** argv) {
    bool isRaw = false, isTrain = false, isPredict = false;
    std::string dataDir, modelOutDir, modelLoadDir, imagePath, inputDir, outputDir, videoPath;
    int imageSize = -1, cameraId = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") isTrain = true;
        else if (arg == "--data-dir" && i + 1 < argc) dataDir = argv[++i];
        else if (arg == "--raw") isRaw = true;
        else if (arg == "--input-dir" && i + 1 < argc) inputDir = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--predict") isPredict = true;
        else if (arg == "--model-dir" && i + 1 < argc) modelLoadDir = argv[++i];
        else if (arg == "--image-size" && i + 1 < argc) imageSize = std::stoi(argv[++i]);
        else if (arg == "--image-path" && i + 1 < argc) imagePath = argv[++i];
        else if (arg == "--video-path" && i + 1 < argc) videoPath = argv[++i];
        else if (arg == "--camera-id" && i + 1 < argc) cameraId = std::stoi(argv[++i]);
    }

    if (isRaw && !inputDir.empty() && !outputDir.empty()) {
        std::filesystem::create_directories(outputDir);
        int maxWidth = imageSize == -1 ? findMaxImageSize(inputDir) : imageSize;
        std::cout << "Max Width: " << maxWidth << std::endl;
        processAndSave(inputDir, outputDir, maxWidth);
        std::cout << "数据已处理并保存到：" << outputDir << std::endl;
        return 0;
    }

    if (isTrain && !dataDir.empty()) {
        std::string modelName = "pca_svm";
        std::string timestamp = getCurrentTimestamp();
        modelOutDir = "models/" + modelName + "_" + timestamp;

        std::filesystem::create_directories(modelOutDir);

        PcaSvmClassifier classifier(100, 5.0, 0.1);
        classifier.buildLabelMapFromDir(dataDir);
        classifier.saveLabelMap(modelOutDir);

        cv::Mat samples, labels;
        loadDataset(dataDir, classifier.getLabelMap(), samples, labels, 250);
        shuffleSamplesAndLabels(samples, labels);

        if (!classifier.train(samples, labels)) {
            std::cerr << "训练失败。" << std::endl;
            return -1;
        }

        if (!classifier.save(modelOutDir)) {
            std::cerr << "模型保存失败。" << std::endl;
            return -1;
        }

        std::cout << "训练完成，模型和标签映射已保存到：" << modelOutDir << std::endl;
        return 0;
    }

    if (isPredict && !modelLoadDir.empty() && imageSize != -1) {
        PcaSvmClassifier classifier;
        if (!classifier.load(modelLoadDir)) {
            std::cerr << "模型加载失败" << std::endl;
            return -1;
        }
        if (!classifier.loadLabelMap(modelLoadDir + "/label_map.txt")) {
            std::cerr << "标签映射加载失败" << std::endl;
            return -1;
        }

        if (!imagePath.empty()) {
            recognizeImage(imagePath, imageSize, classifier);
            return 0;
        } else if (!videoPath.empty()) {
            recognizeVideo(videoPath, imageSize, classifier);
            return 0;
        } else if (cameraId >= 0) {
            recognizeCamera(cameraId, imageSize, classifier);
            return 0;
        }
    }

    std::cerr << "用法:\n"
              << "  数据处理: --raw --input-dir <原始路径> --output-dir <输出路径> [--image-size <尺寸>]\n"
              << "  模型训练: --train --data-dir <处理后图像路径>\n"
              << "  图像识别: --predict --model-dir <模型目录> --image-path <图像路径> --image-size <尺寸>\n"
              << "  视频识别: --predict --model-dir <模型目录> --video-path <视频路径> --image-size <尺寸>\n"
              << "  摄像头识别: --predict --model-dir <模型目录> --camera-id <ID> --image-size <尺寸>\n"
              << std::endl;
    return -1;
}
