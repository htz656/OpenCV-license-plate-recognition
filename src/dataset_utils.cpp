#include "dataset_utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <random>
#include <numeric>
#include <iostream>

#ifdef _WIN32
std::string ws2s(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(wstr);
}
#endif

std::string fixPath(const std::string& path) {
    std::string fixed = path;
    std::replace(fixed.begin(), fixed.end(), '\\', '/');
    return fixed;
}

void loadDataset(const std::string& datasetDir, const std::map<std::string, int>& labelMap,
                 cv::Mat& samples, cv::Mat& labels, int maxPerClass) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto& classDir : std::filesystem::directory_iterator(datasetDir)) {
        if (!classDir.is_directory()) continue;
        auto labelStr = classDir.path().filename().string();
        auto it = labelMap.find(labelStr);
        if (it == labelMap.end()) continue;

        // 收集所有图片路径
        std::vector<std::filesystem::path> imgPaths;
        for (const auto& imgPath : std::filesystem::directory_iterator(classDir)) {
            imgPaths.push_back(imgPath.path());
        }

        // 随机打乱
        std::shuffle(imgPaths.begin(), imgPaths.end(), gen);

        int count = 0;
        for (const auto& imgPath : imgPaths) {
            if (count >= maxPerClass) break;

            std::string pathStr;
#ifdef _WIN32
            pathStr = ws2s(imgPath.wstring());
#else
            pathStr = imgPath.string();
#endif
            pathStr = fixPath(pathStr);

            cv::Mat img = cv::imread(pathStr, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "无法读取文件: " << pathStr << std::endl;
                continue;
            }

            // 强制二值化
            cv::threshold(img, img, 128, 255, cv::THRESH_BINARY);

            img = img.reshape(1, 1);
            img.convertTo(img, CV_32F);

            samples.push_back(img);
            labels.push_back(it->second);
            ++count;
        }
    }
}

void shuffleSamplesAndLabels(cv::Mat& samples, cv::Mat& labels) {
    CV_Assert(samples.rows == labels.rows);
    cv::Mat samplesShuffled(samples.size(), samples.type());
    cv::Mat labelsShuffled(labels.size(), labels.type());

    std::vector<int> idx(samples.rows);
    std::iota(idx.begin(), idx.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(idx.begin(), idx.end(), gen);

    for (int i = 0; i < samples.rows; ++i) {
        samples.row(idx[i]).copyTo(samplesShuffled.row(i));
        labels.row(idx[i]).copyTo(labelsShuffled.row(i));
    }

    samples = samplesShuffled;
    labels = labelsShuffled;
}
