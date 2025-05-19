#include "label_utils.hpp"
#include <filesystem>
#include <fstream>

std::map<std::string, int> buildLabelMap(const std::string& dataDir, std::map<int, std::string>& inverseMap) {
    std::map<std::string, int> labelMap;
    int labelId = 0;
    for (const auto& classDir : std::filesystem::directory_iterator(dataDir)) {
        if (!classDir.is_directory()) continue;
        auto name = classDir.path().filename().string();
        labelMap[name] = labelId;
        inverseMap[labelId++] = name;
    }
    return labelMap;
}

void saveLabelMap(const std::map<std::string, int>& labelMap, const std::string& filePath) {
    std::ofstream ofs(filePath);
    for (const auto& [k, v] : labelMap) ofs << k << " " << v << "\n";
}

std::map<std::string, int> loadLabelMap(const std::string& filePath, std::map<int, std::string>& inverseMap) {
    std::map<std::string, int> labelMap;
    std::ifstream ifs(filePath);
    std::string key; int value;
    while (ifs >> key >> value) {
        labelMap[key] = value;
        inverseMap[value] = key;
    }
    return labelMap;
}
