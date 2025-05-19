#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "PlateLocator.hpp"
#include "dataset_utils.hpp"
#include "image_utils.hpp"
#include "label_utils.hpp"
#include "svm_utils.hpp"

int main(int argc, char** argv) {
    bool isRaw = false, isTrain = false, isPredict = false;
    std::string dataDir, modelOutDir, modelLoadDir, imagePath, inputDir, outputDir;
    int imageSize = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") isTrain = true;
        else if (arg == "--predict") isPredict = true;
        else if (arg == "--raw") isRaw = true;
        else if (arg == "--input-dir" && i + 1 < argc) inputDir = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--data-dir" && i + 1 < argc) dataDir = argv[++i];
        else if (arg == "--model-out" && i + 1 < argc) modelOutDir = argv[++i];
        else if (arg == "--model-dir" && i + 1 < argc) modelLoadDir = argv[++i];
        else if (arg == "--image-size" && i + 1 < argc) imageSize = std::stoi(argv[++i]);
        else if (arg == "--image-path" && i + 1 < argc) imagePath = argv[++i];
    }

    if (isRaw && !inputDir.empty() && !outputDir.empty()) {
        std::filesystem::create_directories(outputDir);
        int maxWidth = imageSize == -1 ? findMaxImageSize(inputDir) : imageSize;
        std::cout << "Max Width: " << maxWidth << std::endl;
        resizeAndPadAndSave(inputDir, outputDir, maxWidth);
        std::cout << "数据已处理并保存到：" << outputDir << std::endl;
        return 0;
    }

    if (isTrain && !dataDir.empty() && !modelOutDir.empty()) {
        std::filesystem::create_directories(modelOutDir);

        std::map<int, std::string> inverseMap;
        auto labelMap = buildLabelMap(dataDir, inverseMap);
        saveLabelMap(labelMap, modelOutDir + "/label_map.txt");

        cv::Mat samples, labels;
        loadDataset(dataDir, labelMap, samples, labels, 250);
        shuffleSamplesAndLabels(samples, labels);

        double minVal, maxVal;
        cv::PCA pca;
        cv::Ptr<cv::ml::SVM> svm = trainPCASVM(samples, labels, pca, 100, minVal, maxVal);

        cv::FileStorage fs(modelOutDir + "/pca.yml", cv::FileStorage::WRITE);
        fs << "mean" << pca.mean;
        fs << "eigenvectors" << pca.eigenvectors;
        fs << "minVal" << minVal;
        fs << "maxVal" << maxVal;
        fs.release();
        svm->save(modelOutDir + "/svm.xml");

        std::cout << "训练完成，模型和标签映射已保存到：" << modelOutDir << std::endl;
        return 0;
    }

    if (!imagePath.empty()) {
        cv::Mat src = cv::imread(imagePath);
        if (src.empty()) {
            std::cerr << "图像加载失败: " << imagePath << std::endl;
            return -1;
        }

        PlateLocator locator;
        cv::Mat resized, preprocessed;
        locator.preprocess(src, resized, preprocessed);

        std::vector<cv::Rect> plates = locator.locatePlates(preprocessed);
        if (plates.empty()) {
            std::cout << "未检测到车牌。" << std::endl;
            return 0;
        }

        cv::Mat plateImg = resized(plates[0]);
        std::vector<cv::Mat> chars = locator.segmentCharacters(plateImg);
        std::cout << "分割出字符数量：" << chars.size() << std::endl;

        if (!isPredict) {
            for (size_t i = 0; i < chars.size(); ++i) {
                cv::imshow("Char " + std::to_string(i), chars[i]);
            }
            cv::waitKey(0);
            return 0;
        }

        if (isPredict && !modelLoadDir.empty() && imageSize != -1) {
            double minVal, maxVal;
            cv::PCA pca;
            cv::FileStorage fs(modelLoadDir + "/pca.yml", cv::FileStorage::READ);
            fs["mean"] >> pca.mean;
            fs["eigenvectors"] >> pca.eigenvectors;
            fs["minVal"] >> minVal;
            fs["maxVal"] >> maxVal;
            fs.release();
            if (pca.mean.empty() || pca.eigenvectors.empty()) {
                std::cerr << "PCA 参数未成功加载" << std::endl;
                return -1;
            }
            auto svm = cv::ml::SVM::load(modelLoadDir + "/svm.xml");

            std::map<int, std::string> inverseMap;
            loadLabelMap(modelLoadDir + "/label_map.txt", inverseMap);

            for (size_t i = 0; i < chars.size(); ++i) {
                cv::Mat resizedChar = resizeToMaxWidth(chars[i], imageSize);
                cv::Mat paddedChar = padToSquareAvgMin(resizedChar, imageSize);
                cv::Mat stretched = stretchGrayPercentile(paddedChar, 0.05, 0.95);
                cv::Mat binaryChar = binarizeByOtsu(stretched, 10);   
                cv::Mat cleaned = removeSmallComponents(binaryChar, 3);
                // cv::imshow("binary", binaryChar);
                // cv::waitKey(0);
                int pred = predictChar(svm, pca, cleaned, minVal, maxVal);
                std::cout << "字符" << i << ": " << inverseMap[pred] << std::endl;
            }
            return 0;
        }
    }

    std::cerr << "用法:\n"
              << "  图像处理: --raw --input-dir <原始路径> --output-dir <输出路径> [--image-size <尺寸>]\n"
              << "  模型训练: --train --data-dir <处理后图像路径> --model-out <保存模型目录> [--image-size <尺寸>]\n"
              << "  模型推理: --predict --model-dir <模型目录> --image-path <图像路径> --image-size <尺寸>\n"
              << "  字符展示: --image-path <图像路径> （仅定位+切割字符，不识别）\n"
              << std::endl;
    return -1;
}
