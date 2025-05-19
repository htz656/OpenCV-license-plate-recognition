#include "image_utils.hpp"
#include <filesystem>

cv::Mat resizeToMinWidth(const cv::Mat& src, int minWidth) {
    int w = src.cols;
    int h = src.rows;
    double scale = static_cast<double>(minWidth) / std::min(w, h);
    int newW = static_cast<int>(w * scale);
    int newH = static_cast<int>(h * scale);

    cv::Mat resized;
    resize(src, resized, cv::Size(newW, newH));
    return resized;
}

cv::Mat resizeToMaxWidth(const cv::Mat& src, int maxWidth) {
    int w = src.cols;
    int h = src.rows;
    double scale = static_cast<double>(maxWidth) / std::max(w, h);
    int newW = static_cast<int>(w * scale);
    int newH = static_cast<int>(h * scale);

    cv::Mat resized;
    resize(src, resized, cv::Size(newW, newH));
    return resized;
}

cv::Mat padToSquareAvgMin(const cv::Mat& src, int width) {
    int top = (width - src.rows) / 2;
    int bottom = width - src.rows - top;
    int left = (width - src.cols) / 2;
    int right = width - src.cols - left;

    double minVal, meanVal;
    cv::minMaxLoc(src, &minVal);
    meanVal = cv::mean(src)[0];
    double padVal = (2 * minVal + meanVal) / 3.0;

    cv::Mat dst;
    cv::copyMakeBorder(src, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(padVal));
    return dst;
}

int findMaxImageSize(const std::string& dataDir) {
    int maxWidth = 0;
    for (const auto& classDir : std::filesystem::directory_iterator(dataDir)) {
        if (!classDir.is_directory()) continue;
        for (const auto& imgPath : std::filesystem::directory_iterator(classDir)) {
            cv::Mat img = cv::imread(imgPath.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            maxWidth = std::max({ maxWidth, img.cols, img.rows });
        }
    }
    return maxWidth;
}

cv::Mat stretchGrayPercentile(const cv::Mat& grayImg, double lowerPercent, double upperPercent) {
    // 1. 计算灰度直方图
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&grayImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // 2. 计算累计分布函数（CDF）
    std::vector<float> cdf(histSize, 0);
    cdf[0] = hist.at<float>(0);
    for (int i = 1; i < histSize; ++i) {
        cdf[i] = cdf[i - 1] + hist.at<float>(i);
    }
    float total = cdf.back();

    // 3. 找到lowerPercent和upperPercent对应的灰度值
    int minGray = 0, maxGray = 255;
    for (int i = 0; i < histSize; ++i) {
        if (cdf[i] / total >= lowerPercent) {
            minGray = i;
            break;
        }
    }
    for (int i = histSize - 1; i >= 0; --i) {
        if (cdf[i] / total <= upperPercent) {
            maxGray = i;
            break;
        }
    }

    // 4. 拉伸到[0, 255]
    cv::Mat stretched;
    grayImg.convertTo(stretched, CV_8UC1, 255.0 / (maxGray - minGray), -minGray * 255.0 / (maxGray - minGray));

    return stretched;
}

cv::Mat binarizeByOtsu(const cv::Mat& grayImg, int offset) {
    cv::Mat binarized;
    double otsuThresh = cv::threshold(grayImg, binarized, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double adjustedThresh = std::min(255.0, otsuThresh + offset);
    cv::threshold(grayImg, binarized, adjustedThresh, 255, cv::THRESH_BINARY);
    return binarized;
}

cv::Mat removeSmallComponents(const cv::Mat& binImg, int minSize) {
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binImg, labels, stats, centroids);
    cv::Mat clean = cv::Mat::zeros(binImg.size(), CV_8UC1);
    for (int i = 1; i < numLabels; ++i) { // i = 0 是背景
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minSize) {
            clean.setTo(255, labels == i);
        }
    }
    return clean;
}

cv::Mat charImgProcess(cv::Mat charImg, int imgeSize) {
    cv::Mat resizedChar = resizeToMaxWidth(charImg, imgeSize);
    cv::Mat paddedChar = padToSquareAvgMin(resizedChar, imgeSize);
    cv::Mat stretched = stretchGrayPercentile(paddedChar, 0.05, 0.95);
    cv::Mat binaryChar = binarizeByOtsu(stretched, 10);
    cv::Mat cleaned = removeSmallComponents(binaryChar, 3);

    return cleaned;
}

void processAndSave(const std::string& dataDir, const std::string& outDir, int imgeSize) {
    for (const auto& classDir : std::filesystem::directory_iterator(dataDir)) {
        if (!classDir.is_directory()) continue;
        auto outClassDir = outDir + "/" + classDir.path().filename().string();
        std::filesystem::create_directories(outClassDir);

        for (const auto& imgPath : std::filesystem::directory_iterator(classDir)) {
            cv::Mat img = cv::imread(imgPath.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            cv::Mat processedImg = charImgProcess(img, imgeSize);

            cv::imwrite(outClassDir + "/" + imgPath.path().filename().string(), processedImg);
        }
    }
}
