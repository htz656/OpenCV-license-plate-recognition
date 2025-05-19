#include "svm_utils.hpp"

cv::Ptr<cv::ml::SVM> trainPCASVM(const cv::Mat& samples, const cv::Mat& labels,
                                 cv::PCA& pca, int numComponents,
                                 double& minVal, double& maxVal) {
    // 计算归一化参数
    cv::minMaxLoc(samples, &minVal, &maxVal);

    // 归一化
    cv::Mat samplesNorm;
    samples.convertTo(samplesNorm, CV_32F, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

    // PCA 降维
    pca = cv::PCA(samplesNorm, cv::Mat(), cv::PCA::DATA_AS_ROW, numComponents);
    cv::Mat samplesPCA;
    pca.project(samplesNorm, samplesPCA);

    // SVM 训练
    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(0.1);
    svm->setC(5.0);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100000, 1e-6));
    svm->train(samplesPCA, cv::ml::ROW_SAMPLE, labels);

    return svm;
}

int predictChar(cv::Ptr<cv::ml::SVM> svm, const cv::PCA& pca,
                const cv::Mat& binaryCharImg, double minVal, double maxVal) {
    cv::Mat sample = binaryCharImg.reshape(1, 1);
    sample.convertTo(sample, CV_32F);
    sample = (sample - minVal) / (maxVal - minVal);
    
    cv::Mat samplePCA;
    pca.project(sample, samplePCA);

    int pred = static_cast<int>(svm->predict(samplePCA));
    return pred;
}
