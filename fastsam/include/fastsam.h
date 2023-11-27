#pragma once

#include "inference.hpp"
#include <omp.h>

class FastSam : public Inference
{
public:
    explicit FastSam(const std::string &engine_file_path, const int warm_cnt = 5);

    ~FastSam() = default;

    void run(const Mat &, vector<Object> &);

    static void draw_objects(const Mat &image, Mat &res, const std::vector<Object> &objs);

protected:
    void copy_from_Mat(const cv::Mat &image);

    void copy_from_Mat(const cv::Mat &image, cv::Size &size);

    void letterbox(
        const Mat &image,
        Mat &out,
        cv::Size &size);

    void postprocess(vector<Object> &objs);

    PreParam pparam;

private:
    cv::Size _input_size;
    float _score_thres;
    float _mask_conf;
    float _iou_thres;
    int _topk;
};
