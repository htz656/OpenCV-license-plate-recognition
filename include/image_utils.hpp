#pragma once

#include <opencv2/opencv.hpp>

cv::Mat resizeToMinWidth(const cv::Mat& src, int minWidth);
cv::Mat resizeToMaxWidth(const cv::Mat& src, int maxWidth);
int findMaxImageSize(const std::string& dataDir);
cv::Mat charImgProcess(cv::Mat charImg, int imgeSize);
void processAndSave(const std::string& dataDir, const std::string& outDir, int maxWidth);
