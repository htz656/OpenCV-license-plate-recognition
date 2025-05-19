#include "model.hpp"
#include <filesystem>
#include <fstream>

PcaSvmClassifier::PcaSvmClassifier(int numComponents_, double svmC_, double svmGamma_, int epochs_)
    : numComponents(numComponents_), svmC(svmC_), svmGamma(svmGamma_), epochs(epochs_),
      minVal(0.0), maxVal(255.0) {}

void PcaSvmClassifier::setNormalizationRange(double minV, double maxV) {
    minVal = minV;
    maxVal = maxV;
}

bool PcaSvmClassifier::train(const cv::Mat& samples, const cv::Mat& labels) {
    cv::minMaxLoc(samples, &minVal, &maxVal);
    if (maxVal - minVal < 1e-6) return false;

    cv::Mat samplesNorm;
    samples.convertTo(samplesNorm, CV_32F, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

    pca = cv::PCA(samplesNorm, cv::Mat(), cv::PCA::DATA_AS_ROW, numComponents);
    cv::Mat samplesPCA;
    pca.project(samplesNorm, samplesPCA);

    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(svmGamma);
    svm->setC(svmC);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, epochs, 1e-6));

    return svm->train(samplesPCA, cv::ml::ROW_SAMPLE, labels);
}

int PcaSvmClassifier::predict(const cv::Mat& processedCharImage) const {
    if (svm.empty() || pca.eigenvectors.empty()) return -1;

    cv::Mat sample = processedCharImage.reshape(1, 1);
    sample.convertTo(sample, CV_32F);
    sample = (sample - minVal) / (maxVal - minVal);

    cv::Mat samplePCA;
    pca.project(sample, samplePCA);

    return static_cast<int>(svm->predict(samplePCA));
}

bool PcaSvmClassifier::save(const std::string& dirPath) const {
    if (!svm || pca.eigenvectors.empty()) return false;
    std::filesystem::create_directories(dirPath);

    cv::FileStorage fs(dirPath + "/pca.yml", cv::FileStorage::WRITE);
    fs << "mean" << pca.mean;
    fs << "eigenvectors" << pca.eigenvectors;
    fs << "minVal" << minVal;
    fs << "maxVal" << maxVal;
    fs << "numComponents" << numComponents;
    fs << "svmC" << svmC;
    fs << "svmGamma" << svmGamma;
    fs.release();

    svm->save(dirPath + "/svm.xml");
    return true;
}

bool PcaSvmClassifier::load(const std::string& dirPath) {
    cv::FileStorage fs(dirPath + "/pca.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    fs["mean"] >> pca.mean;
    fs["eigenvectors"] >> pca.eigenvectors;
    fs["minVal"] >> minVal;
    fs["maxVal"] >> maxVal;
    fs["numComponents"] >> numComponents;
    fs["svmC"] >> svmC;
    fs["svmGamma"] >> svmGamma;
    fs.release();

    svm = cv::ml::SVM::load(dirPath + "/svm.xml");
    return !svm.empty();
}

void PcaSvmClassifier::buildLabelMapFromDir(const std::string& dataDir) {
    labelMap.clear();
    inverseMap.clear();
    int labelId = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dataDir)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        labelMap[name] = labelId;
        inverseMap[labelId] = name;
        ++labelId;
    }
}

bool PcaSvmClassifier::saveLabelMap(const std::string& dirPath) const {
    std::ofstream ofs(dirPath  + "/label_map.txt");
    if (!ofs.is_open()) return false;
    for (const auto& [k, v] : labelMap) {
        ofs << k << " " << v << "\n";
    }
    return true;
}

bool PcaSvmClassifier::loadLabelMap(const std::string& dirPath) {
    labelMap.clear();
    inverseMap.clear();
    std::ifstream ifs(dirPath);
    if (!ifs.is_open()) return false;
    std::string key;
    int value;
    while (ifs >> key >> value) {
        labelMap[key] = value;
        inverseMap[value] = key;
    }
    return true;
}

int PcaSvmClassifier::labelToId(const std::string& label) const {
    auto it = labelMap.find(label);
    return it == labelMap.end() ? -1 : it->second;
}

std::string PcaSvmClassifier::idToLabel(int id) const {
    auto it = inverseMap.find(id);
    return it == inverseMap.end() ? "" : it->second;
}