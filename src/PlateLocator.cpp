#include "PlateLocator.hpp"

PlateLocator::PlateLocator(
    int targetMaxWidth, 
    int targetMaxHeight,
    cv::Size blurKernelSize, 
    double gammaValue, 
    int radius,
    int cannyThreshold1, 
    int cannyThreshold2,
    cv::Size morphKernel1Size, 
    cv::Size morphKernel2Size
) : maxWidth(targetMaxWidth), 
    maxHeight(targetMaxHeight),
    blurKernel(blurKernelSize), 
    gamma(gammaValue), 
    radius(radius),
    canny1(cannyThreshold1), 
    canny2(cannyThreshold2),
    kernel1Size(morphKernel1Size), 
    kernel2Size(morphKernel2Size) {}

void PlateLocator::preprocess(const cv::Mat& origin, cv::Mat& resized, cv::Mat& preprocessed) const {
    cv::Mat resizedImg, grayImg, blurImg, stretchGrayImg, openImg, diffImg, binaryImg, edgeImg;
    cv::Mat closeImg1, closeImg2, openImg1, openImg2;

    double scale = std::min(static_cast<double>(maxWidth) / origin.cols, static_cast<double>(maxHeight) / origin.rows);
    cv::resize(origin, resizedImg, cv::Size(), scale, scale, cv::INTER_LINEAR);
    resized = resizedImg.clone();

    cv::cvtColor(resizedImg, grayImg, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImg, blurImg, blurKernel, 0);

    cv::Mat normImg, gammaImg;
    blurImg.convertTo(normImg, CV_32F, 1.0 / 255.0);
    cv::pow(normImg, gamma, gammaImg);
    gammaImg.convertTo(stretchGrayImg, CV_8U, 255.0);

    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(static_cast<int>(3.14 * radius), radius));
    cv::morphologyEx(stretchGrayImg, openImg, cv::MORPH_OPEN, rectKernel);
    cv::absdiff(stretchGrayImg, openImg, diffImg);

    cv::threshold(diffImg, binaryImg, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::Canny(binaryImg, edgeImg, canny1, canny2);

    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, kernel1Size);
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, kernel2Size);
    cv::morphologyEx(edgeImg, closeImg1, cv::MORPH_CLOSE, kernel1);
    cv::morphologyEx(closeImg1, openImg1, cv::MORPH_OPEN, kernel2);
    cv::morphologyEx(openImg1, closeImg2, cv::MORPH_CLOSE, kernel1);
    cv::morphologyEx(closeImg2, openImg2, cv::MORPH_OPEN, kernel2);

    preprocessed = openImg2.clone();
}

std::vector<cv::Rect> PlateLocator::locatePlates(
    const cv::Mat& preprocessedImg,
    float minAspectRatio,
    float maxAspectRatio,
    float targetAspectRatio,
    float minRectAreaRatio,
    float maxRectAreaRatio,
    float minFillRatio,
    int remain
) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(preprocessedImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> plateRects;

    float totalArea = static_cast<float>(preprocessedImg.cols * preprocessedImg.rows);
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        float w = rect.width, h = rect.height;
        if (w == 0 || h == 0) continue;

        float aspectRatio = w / h;
        float areaRatio = (w * h) / totalArea;
        if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) continue;
        if (areaRatio < minRectAreaRatio || areaRatio > maxRectAreaRatio) continue;

        cv::Mat roi = preprocessedImg(rect);
        float fillRatio = static_cast<float>(cv::countNonZero(roi)) / (w * h);
        if (fillRatio < minFillRatio) continue;

        plateRects.push_back(rect);
    }

    std::sort(plateRects.begin(), plateRects.end(),
        [targetAspectRatio](const cv::Rect& a, const cv::Rect& b) {
            float aspectA = static_cast<float>(a.width) / a.height;
            float aspectB = static_cast<float>(b.width) / b.height;
            return std::abs(aspectA - targetAspectRatio) < std::abs(aspectB - targetAspectRatio);
        });

    if (plateRects.size() > remain) plateRects.resize(remain);
    return plateRects;
}
