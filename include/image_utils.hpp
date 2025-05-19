#pragma once

#include <opencv2/opencv.hpp>

cv::Mat resizeToMinWidth(const cv::Mat& src, int minWidth);
cv::Mat resizeToMaxWidth(const cv::Mat& src, int maxWidth);
cv::Mat padToSquareAvgMin(const cv::Mat& src, int width);
int findMaxImageSize(const std::string& dataDir);
cv::Mat stretchGrayPercentile(const cv::Mat& grayImg, double lowerPercent = 0.01, double upperPercent = 0.99);
cv::Mat binarizeByOtsu(const cv::Mat& grayImg, int offset = 10);
cv::Mat removeSmallComponents(const cv::Mat& binImg, int minSize);
void resizeAndPadAndSave(const std::string& dataDir, const std::string& outDir, int maxWidth);
