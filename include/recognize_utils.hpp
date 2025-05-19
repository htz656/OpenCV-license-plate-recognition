#pragma once

#include <string>
#include "model.hpp"

void recognizeImage(const std::string& imagePath, int imageSize, PcaSvmClassifier& classifier);
void recognizeVideo(const std::string& videoPath, int imageSize, PcaSvmClassifier& classifier);
void recognizeCamera(int cameraId, int imageSize, PcaSvmClassifier& classifier);
