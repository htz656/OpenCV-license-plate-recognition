#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <string>

void loadDataset(const std::string& datasetDir, const std::map<std::string, int>& labelMap,
                 cv::Mat& samples, cv::Mat& labels, int maxPerClass = 100);
void shuffleSamplesAndLabels(cv::Mat& samples, cv::Mat& labels);
