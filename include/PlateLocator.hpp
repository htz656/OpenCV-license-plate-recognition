#ifndef PLATE_LOCATOR_H
#define PLATE_LOCATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class PlateLocator {
public:
    PlateLocator(
        int targetMaxWidth = 1024,
        int targetMaxHeight = 720,
        cv::Size blurKernelSize = cv::Size(5, 5),
        double gammaValue = 0.2,
        int radius = 15,
        int cannyThreshold1 = 100,
        int cannyThreshold2 = 200,
        cv::Size morphKernel1Size = cv::Size(44, 14),
        cv::Size morphKernel2Size = cv::Size(9, 4)
    );

    void preprocess(
        const cv::Mat& origin, 
        cv::Mat& resized, 
        cv::Mat& preprocessed
    ) const;

    std::vector<cv::Rect> locatePlates(
        const cv::Mat& preprocessedImg,
        float minAspectRatio = 2.1f,
        float maxAspectRatio = 4.2f,
        float targetAspectRatio = 3.14f,
        float minRectAreaRatio = 0.005f,
        float maxRectAreaRatio = 0.5f,
        float minFillRatio = 0.5f,
        int remain = 3
    ) const;

    std::vector<cv::Mat> segmentCharacters(
        const cv::Mat& plateImg
    ) const;

private:
    int maxWidth, maxHeight;
    cv::Size blurKernel;
    double gamma;
    int radius;
    int canny1, canny2;
    cv::Size kernel1Size, kernel2Size;
};

#endif // PLATE_LOCATOR_H
