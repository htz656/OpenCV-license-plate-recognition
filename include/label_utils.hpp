#pragma once

#include <map>
#include <string>

std::map<std::string, int> buildLabelMap(const std::string& dataDir, std::map<int, std::string>& inverseMap);
void saveLabelMap(const std::map<std::string, int>& labelMap, const std::string& filePath);
std::map<std::string, int> loadLabelMap(const std::string& filePath, std::map<int, std::string>& inverseMap);
