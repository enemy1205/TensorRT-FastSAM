//
// Created by saika on 10/30/23.
//
#include "chrono"
#include "cmdline.h"
#include "opencv2/opencv.hpp"
#include "fastsam.h"

int main(int argc, char **argv)
{
    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to engine model.", true,
                         "fast_sam_s.engine");
    cmd.add<std::string>("input_path", 'i', "Input source to be detected.",
                         true);
    cmd.add<std::string>("save_path", 's', "The path where result image save",
                         true);
    cmd.add<int>("cuda", 'c', "Path to class names file.", false, 0);

    cmd.parse_check(argc, argv);
    cudaSetDevice(cmd.get<int>("cuda"));
    const std::string engine_file_path{cmd.get<std::string>("model_path")};
    string input_path = cmd.get<std::string>("input_path");
    string save_path = cmd.get<std::string>("save_path");

    auto fastsam = new FastSam(engine_file_path);
    bool isVideo{false};

    if (IsFile(input_path))
    {
        std::string suffix = input_path.substr(input_path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            isVideo = false;
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" ||
                 suffix == "mpeg" || suffix == "mov" || suffix == "mkv")
        {
            isVideo = true;
        }
        else
        {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else
    {
        printf("Please input a file");
        std::abort();
    }
    cv::Mat color_mask, image;
    // The output is all in the `objs` object
    std::vector<Object> objs;
    if (isVideo)
    {
        cv::VideoCapture cap(input_path);

        if (!cap.isOpened())
        {
            printf("can not open %s\n", input_path.c_str());
            return -1;
        }
        // 获取视频文件的参数
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter videoWriter(save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frameWidth, frameHeight));
        int cnt = 0;
        double time_sum = 0;
        while (cap.read(image))
        {
            objs.clear();
            auto start = std::chrono::system_clock::now();
            fastsam->run(image, objs);
            fastsam->draw_objects(image, color_mask, objs);
            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                          end - start)
                          .count() /
                      1000.;
            printf("cost %2.4lf ms\n", tc);
            videoWriter.write(color_mask);
            ++cnt;
            time_sum += tc;
        }
        printf("cost average: %2.4lf ms\n", time_sum / cnt);
    }
    else
    {
        objs.clear();
        image = cv::imread(input_path);
        auto start = std::chrono::system_clock::now();
        fastsam->run(image, objs);
        fastsam->draw_objects(image, color_mask, objs);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      end - start)
                      .count() /
                  1000.;
        printf("cost %2.4lf ms\n", tc);
        cv::imwrite(save_path, color_mask);
    }
    delete fastsam;
    return 0;
}