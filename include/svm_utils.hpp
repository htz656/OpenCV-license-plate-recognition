#pragma once

#include <opencv2/opencv.hpp>

cv::Ptr<cv::ml::SVM> trainPCASVM(const cv::Mat& samples, const cv::Mat& labels,
                                 cv::PCA& pca, int numComponents,
                                 double& minVal, double& maxVal) ;
int predictChar(cv::Ptr<cv::ml::SVM> svm, const cv::PCA& pca,
                const cv::Mat& binaryCharImg, double minVal, double maxVal);
