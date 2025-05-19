# OpenCV License Plate Recognition

本项目基于 OpenCV 实现车牌识别，支持图像、视频和摄像头输入，结合字符分割与 PCA+SVM 分类器完成字符识别。

## 功能特性

* 车牌定位
* 字符分割
* 字符识别
* 支持图像、视频文件与摄像头输入

## 项目结构

```
OpenCV-license-plate-recognition/
├── .vscode/
├── CMakeLists.txt
├── include/
├── src/
│   ├── main.cpp                # 主程序入口
│   ├── model.cpp               # PCA+SVM 分类器类
│   ├── PlateLocator.cpp        # 车牌定位与字符分割
│   ├── dataset_utils.cpp       # 字符识别数据集加载
│   ├── image_utils.cpp         # 图片处理相关函数
│   └── recognize_utils.cpp     # 图像/视频/摄像头识别逻辑
├── example/                    # 测试使用示例图片
├── dataset/                    # 字符图像数据集
├── models/                     # 训练生成的模型及labelMap
├── build/                      # 构建输出
├── LICENSE
├── .gitignore
└── README.md
```

## 字符识别数据集来源与说明

本项目所使用的训练数据来源于 [EasyPR 项目](https://github.com/liuruoze/EasyPR/tree/master/resources/train) 提供的数据集：

* 🔹 [annCh.7z](https://github.com/liuruoze/EasyPR/blob/master/resources/train/annCh.7z)
* 🔹 [annGray.7z](https://github.com/liuruoze/EasyPR/blob/master/resources/train/annGray.7z)

数据解压并经过统一预处理，已整理至 `dataset/` 目录，供模型训练使用。

数据集遵循 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。


## 项目使用说明
该项目支持三种主要功能：数据预处理、模型训练 和 模型预测（图像、视频、摄像头）。通过命令行参数指定操作类型和相关路径。

### 1. 数据预处理
用于将 EasyPR 提供的原始字符图像转换为统一尺寸的二值图，用于训练或预测。

```bash
./main --raw --input-dir dataset/raw --output-dir dataset/processed --image-size 20
```

参数说明：
- --raw：启用数据预处理功能。
- --input-dir：原始图像目录（例如：annCh 或 annGray 解压后的路径）。
- --output-dir：处理后图像保存路径。
- --image-size（可选）：统一缩放到的图像宽度（默认会自动检测最大宽度）。


### 2. 模型训练（PCA + SVM）
从处理好的图像中提取特征，进行 PCA 降维 + SVM 分类器训练，并保存模型。

```bash
./main --train --data-dir dataset/processed
```

参数说明：
- --train：启用训练模式。
- --data-dir：预处理后的图像路径（按类名分类子文件夹）。

模型输出路径为 models/pca_svm_年月日时分秒/，包含模型文件和标签映射表。

### 3. 模型预测
#### 图像识别
```bash
./main --predict --model-dir models/pca_svm_xxxxx --image-path path/to/image.jpg --image-size 20
```

#### 视频识别
```bash
./main --predict --model-dir models/pca_svm_xxxxx --video-path path/to/video.mp4 --image-size 20
```

#### 摄像头识别
```bash
./main --predict --model-dir models/pca_svm_xxxxx --camera-id 0 --image-size 250
```

通用参数说明：
- --predict：启用预测模式。
- --model-dir：已训练模型的目录（包含 SVM 模型和 label_map.txt）。
- --image-path / --video-path / --camera-id：输入类型三选一。
- --image-size：字符图像大小应与训练时保持一致。

## License

本项目源代码采用 MIT 许可证发布，训练数据遵循 Apache License 2.0。
