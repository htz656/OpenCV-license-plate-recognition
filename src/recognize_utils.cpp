#include "recognize_utils.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include "PlateLocator.hpp"
#include "image_utils.hpp"

cv::Mat processFrame(const cv::Mat& src, int imgSize, PcaSvmClassifier& classifier) {
    PlateLocator locator;
    cv::Mat resized, preprocessed;
    locator.preprocess(src, resized, preprocessed);
    cv::Mat drawImg = resized.clone();
    std::vector<cv::Rect> plates = locator.locatePlates(preprocessed);
    if (plates.empty()) {
        std::cout << "处理失败或未检测到车牌" << std::endl;
        return drawImg;
    }

    cv::Mat plateImg = resized(plates[0]);
    std::vector<cv::Mat> chars = locator.segmentCharacters(plateImg);
    std::cout << "分割出字符数量：" << chars.size() << std::endl;

    std::string plateText;
    for (size_t i = 0; i < chars.size(); ++i) {
        cv::Mat processedImg = charImgProcess(chars[i], imgSize);

        int pred = classifier.predict(processedImg);
        std::string label = classifier.idToLabel(pred);
        plateText += label;
    }

    std::cout << "车牌号: " + plateText << std::endl;

    cv::Rect plateRect = plates[0];
    cv::rectangle(drawImg, plateRect, cv::Scalar(0, 255, 0), 2);

    // 在车牌框上方标注识别出的车牌号
    int baseline = 0;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    int thickness = 2;
    cv::Size textSize = cv::getTextSize(plateText, font, fontScale, thickness, &baseline);
    cv::Point textOrg(plateRect.x, plateRect.y - 5); // 文字位置：车牌框上方

    // 防止文字越界到图像外
    if (textOrg.y < textSize.height) {
        textOrg.y = plateRect.y + textSize.height + 5;
    }
    cv::putText(drawImg, plateText, textOrg, font, fontScale, cv::Scalar(0, 0, 255), thickness);

    return drawImg;
}

void recognizeImage(const std::string& imagePath, int imgSize, PcaSvmClassifier& classifier) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "图像加载失败: " << imagePath << std::endl;
        return;
    }
    cv::Mat drawFrame = processFrame(img, imgSize, classifier);
    if (drawFrame.empty()) {
        std::cout << "处理失败或未检测到车牌" << std::endl;
    } else {
        cv::imshow("车牌识别结果", drawFrame);
        cv::waitKey(0);
    }
}

void recognizeVideo(const std::string& videoPath, int imgSize, PcaSvmClassifier& classifier) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << videoPath << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat drawImg = processFrame(frame, imgSize, classifier);
        cv::imshow("Video Frame", drawImg);
        if (cv::waitKey(30) == 27) break;
    }
}

void recognizeCamera(int cameraId, int imgSize, PcaSvmClassifier& classifier) {
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头: " << cameraId << std::endl;
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat drawImg = processFrame(frame, imgSize, classifier);
        cv::imshow("Camera", drawImg);
        if (cv::waitKey(30) == 27) break;
    }
}
