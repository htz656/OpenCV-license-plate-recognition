#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define PWIDTH 1024
#define PHEIGHT 720

#define PLATE_TARGET_WIDTH 220
#define PLATE_TARGET_HEIGHT 70

void preprocess(
    const cv::Mat& origin,
    cv::Mat& resized,
    cv::Mat& preprocessed,
    int targetMaxWidth = PWIDTH,
    int targetMaxHeight = PHEIGHT,
    cv::Size blurKernelSize = cv::Size(5, 5),
    double gammaValue = 0.2,
    int radius = 15,
    int cannyThreshold1 = 100,
    int cannyThreshold2 = 200,
    cv::Size morphKernel1Size = cv::Size(44, 14),
    cv::Size morphKernel2Size = cv::Size(9, 4.5)
) {
    cv::Mat resizedImg, grayImg, blurImg, stretchGrayImg, openImg, diffImg, binaryImg, edgeImg;
    cv::Mat closeImg1, closeImg2, openImg1, openImg2;

    // 缩放图像
    int originImageWidth = origin.cols;
    int originImageHeight = origin.rows;
    double scaleW = static_cast<double>(targetMaxWidth) / originImageWidth;
    double scaleH = static_cast<double>(targetMaxHeight) / originImageHeight;
    double scale = std::min(scaleW, scaleH);
    int newWidth = static_cast<int>(originImageWidth * scale);
    int newHeight = static_cast<int>(originImageHeight * scale);
    cv::resize(origin, resizedImg, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    resized = resizedImg.clone();  

    // 灰度与高斯模糊
    cv::cvtColor(resizedImg, grayImg, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImg, blurImg, blurKernelSize, 0);

    // 非线性拉伸（Gamma校正）
    // double gammaValue = 0.5;  // 可调，<1 增强暗部细节
    cv::Mat normImg, gammaImg;
    blurImg.convertTo(normImg, CV_32F, 1.0 / 255.0); // 归一化
    cv::pow(normImg, gammaValue, gammaImg);          // 非线性变换
    gammaImg.convertTo(stretchGrayImg, CV_8U, 255.0); // 反归一化

    // 顶帽操作
    cv::Mat rectKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3.14 * radius, radius));
    cv::morphologyEx(stretchGrayImg, openImg, cv::MORPH_OPEN, rectKernel);
    cv::absdiff(stretchGrayImg, openImg, diffImg);

    // 二值化
    cv::threshold(diffImg, binaryImg, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    // 边缘检测
    cv::Canny(binaryImg, edgeImg, cannyThreshold1, cannyThreshold2);

    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, morphKernel1Size);
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, morphKernel2Size);
    cv::morphologyEx(edgeImg, closeImg1, cv::MORPH_CLOSE, kernel1);
    cv::morphologyEx(closeImg1, openImg1, cv::MORPH_OPEN, kernel2);
    cv::morphologyEx(openImg1, closeImg2, cv::MORPH_CLOSE, kernel1);
    cv::morphologyEx(closeImg2, openImg2, cv::MORPH_OPEN, kernel2);

    // 输出
    preprocessed = openImg2.clone();
}

std::vector<cv::Rect> locatePotentialPlate(
    const cv::Mat& preprocessedImg,
    float minAspectRatio = 2.1f,
    float maxAspectRatio = 4.2f,
    float targetAspectRatio = 3.14f,
    float minRectAreaRatio = 0.005f,
    float maxRectAreaRatio = 0.5f,
    float minFillRatio = 0.5f,
    int remain = 3
) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(preprocessedImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> plateRects;

    float totalImageArea = static_cast<float>(preprocessedImg.cols * preprocessedImg.rows);

    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        float w = rect.width;
        float h = rect.height;

        if (w == 0 || h == 0) continue;

        float aspectRatio = static_cast<float>(w) / h;
        float areaRatio = w * h / totalImageArea;

        if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) continue;
        if (areaRatio < minRectAreaRatio || areaRatio > maxRectAreaRatio) continue;

        cv::Mat roi = preprocessedImg(rect);
        int whitePixels = cv::countNonZero(roi);
        float fillRatio = static_cast<float>(whitePixels) / (w * h);
        if (fillRatio < minFillRatio) continue;

        plateRects.push_back(rect);
    }

    // 按照长宽比接近 targetAspectRatio 排序
    std::sort(plateRects.begin(), plateRects.end(),
        [targetAspectRatio](const cv::Rect& a, const cv::Rect& b) {
            float aspectA = static_cast<float>(a.width) / a.height;
            float aspectB = static_cast<float>(b.width) / b.height;
            return std::abs(aspectA - targetAspectRatio) < std::abs(aspectB - targetAspectRatio);
        });

    if (plateRects.size() > remain) {
        plateRects.resize(remain);
    }

    return plateRects;
}



int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <图像路径>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) {
        std::cout << "图像加载失败: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat resized, preprocessed;
    preprocess(src, resized, preprocessed);
    bool success = cv::imwrite("data/preprocessed.jpg", preprocessed);
    if (success) {
        std::cout << "图像已成功保存为 preprocessed.jpg" << std::endl;
    } else {
        std::cout << "图像保存失败" << std::endl;
    }

    std::vector<cv::Rect> potentialPlates = locatePotentialPlate(preprocessed);

    cv::Mat drawImg = resized.clone();
    if (!potentialPlates.empty()) {
        std::cout << "Found " << potentialPlates.size() << " potential plate(s)." << std::endl;

        cv::Rect bestPlateRect = potentialPlates[0];
        cv::rectangle(drawImg, bestPlateRect, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Detected Plates", drawImg);
    } else {
        std::cout << "No plates found." << std::endl;
        cv::imshow("No Plates Found", drawImg);
    }
    
    cv::imshow("Preprocessed", preprocessed);
    cv::waitKey(0);
    return 0;
}
