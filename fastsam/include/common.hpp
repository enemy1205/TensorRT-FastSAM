#pragma once

#include <sys/stat.h>
#include <unistd.h>
#include "NvInfer.h"
#include <dirent.h>

#include <array>
#include <fstream>
#include <opencv2/opencv.hpp>
using std::cout, std::endl,std::string,std::vector;


 inline int scanFiles(vector<string> &fileList, string &inputDirectory,
              const std::function<bool(const string &)> &condition_func) {
    if (reinterpret_cast<const char *>(inputDirectory.back()) != "/")
        inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char *str = inputDirectory.c_str();

    p_dir = opendir(str);
    if (p_dir == nullptr) {
        cout << "can't open :" << inputDirectory << endl;
    }

    struct dirent *p_dirent;

    while ((p_dirent = readdir(p_dir))) {
        string tmpFileName = p_dirent->d_name;
        if (tmpFileName.length() < 4 || condition_func(tmpFileName)) {
            continue;
        } else {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList.size();
}

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

struct Object {
    cv::Rect_<float> rect;
    [[maybe_unused]]  int label = 0;  
    float prob = 0.0;
    cv::Mat boxMask;
};

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};

struct FaceBbox {
    cv::Rect rect;
    float score;
    std::array<float, 10> landmarks;
};
struct anchorBox {
    float c_x;
    float c_y;
    float width;
    float height;
};

// 左闭右闭区间
inline int getRand(int min, int max) {
    return (rand() % (max - min + 1)) + min;
}

inline bool fileExists(const std::string& filename) {
    std::ifstream file(filename.c_str());
    return file.good();
}

inline bool IsPathExist(const std::string &path) {
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string &path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string &path) {
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

inline int get_size_by_dims(const nvinfer1::Dims &dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType &dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

#define CHECK(call)                                                   \
  do {                                                                \
    const cudaError_t error_code = call;                              \
    if (error_code != cudaSuccess) {                                  \
      printf("CUDA Error:\n");                                        \
      printf("    File:       %s\n", __FILE__);                       \
      printf("    Line:       %d\n", __LINE__);                       \
      printf("    Error code: %d\n", error_code);                     \
      printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

class Logger : public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(
            nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
            : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity,
             const char *msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};