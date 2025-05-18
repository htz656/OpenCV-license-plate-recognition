#include <opencv2/opencv.hpp>
#include <iostream>
#include "PlateLocator.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <图像路径>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) {
        std::cerr << "图像加载失败: " << imagePath << std::endl;
        return -1;
    }

    PlateLocator locator;
    cv::Mat resized, preprocessed;
    locator.preprocess(src, resized, preprocessed);

    cv::imwrite("data/preprocessed.jpg", preprocessed);

    std::vector<cv::Rect> plates = locator.locatePlates(preprocessed);

    cv::Mat drawImg = resized.clone();
    if (!plates.empty()) {
        std::cout << "发现 " << plates.size() << " 个候选车牌区域。" << std::endl;
        cv::rectangle(drawImg, plates[0], cv::Scalar(0, 255, 0), 2);
        cv::imshow("Detected Plates", drawImg);
    } else {
        std::cout << "未检测到车牌。" << std::endl;
        cv::imshow("No Plates Found", drawImg);
    }

    cv::imshow("Preprocessed", preprocessed);
    cv::waitKey(0);
    return 0;
}
