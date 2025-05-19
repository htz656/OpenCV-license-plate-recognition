#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class PcaSvmClassifier {
public:
    PcaSvmClassifier(int numComponents = 100,
                     double svmC = 5.0,
                     double svmGamma = 0.1,
                     int epochs = 100000);

    bool train(const cv::Mat& samples, const cv::Mat& labels);
    int predict(const cv::Mat& binaryCharImage) const;

    bool save(const std::string& dirPath) const;
    bool load(const std::string& dirPath);

    void setNormalizationRange(double minV, double maxV);
    double getMinVal() const { return minVal; }
    double getMaxVal() const { return maxVal; }

    void buildLabelMapFromDir(const std::string& dataDir);
    bool saveLabelMap(const std::string& filePath) const;
    bool loadLabelMap(const std::string& filePath);
    std::map<std::string, int> getLabelMap() const {return labelMap;};
    std::map<int, std::string> getInverseLabelMap() const {return inverseMap;};

    int labelToId(const std::string& label) const;
    std::string idToLabel(int id) const;

private:
    int numComponents, epochs;
    double svmC, svmGamma;
    double minVal, maxVal;

    cv::PCA pca;
    cv::Ptr<cv::ml::SVM> svm;

    std::map<std::string, int> labelMap;
    std::map<int, std::string> inverseMap;
};
