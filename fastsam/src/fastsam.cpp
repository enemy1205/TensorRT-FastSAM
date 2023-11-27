
#include "fastsam.h"

/**
 * @brief 依据长边等比例缩放,短边部分填充
 * @param image 输入图像
 * @param out 输出图像
 * @param size 缩放至尺寸
 */
void FastSam::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else
    {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                       {0, 0, 0});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0),
                           true, false, CV_32F);
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

/**
 * @brief 输入图像预处理,序列化,转移至显存
 * @param image
 */
void FastSam::copy_from_Mat(const cv::Mat &image)
{
    cv::Mat nchw;
    auto &in_binding = this->input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0,
                                        nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                          nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                          this->stream));
}

/**
 * @brief 输入图像预处理,序列化,转移至显存
 * @param image
 * @param size 指定输入尺寸
 */
void FastSam::copy_from_Mat(const cv::Mat &image, cv::Size &size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(
        0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(),
                          nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                          this->stream));
}

/**
 * @brief 分割解码后处理
 * @param objs 所有对象数组
 */
void FastSam::postprocess(std::vector<Object> &objs)
{
    objs.clear();
    auto input_h = this->input_bindings[0].dims.d[2];
    auto input_w = this->input_bindings[0].dims.d[3];
    int mask_channel = this->output_bindings[0].dims.d[1];
    int mask_h = this->output_bindings[0].dims.d[2];
    int mask_w = this->output_bindings[0].dims.d[3];
    int class_number = 1;
    int mask_number = this->output_bindings[5].dims.d[1] - class_number - 4;
    int candidates = this->output_bindings[5].dims.d[2];
    // this->output_bindings[5].dims.d 1*37*21504
    // 37 = 4+1+32 box+conf+mask

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    auto &dw = this->pparam.dw;
    auto &dh = this->pparam.dh;
    auto &width = this->pparam.width;
    auto &height = this->pparam.height;
    auto &ratio = this->pparam.ratio;

    auto *output = static_cast<float *>(this->host_ptrs[5]);
    cv::Mat protos = cv::Mat(mask_channel, mask_h * mask_w, CV_32F,
                             static_cast<float *>(this->host_ptrs[0]));
    for (size_t i = 0; i < candidates; ++i)
    {
        float score = *(output + 4 * candidates + i);
        if (score > _score_thres)
        {
            // center_x, center_y, width, height
            float w = *(output + 2 * candidates + i);
            float h = *(output + 3 * candidates + i);

            float x0 = *(output + 0 * candidates + i) - dw - w / 2;
            float y0 = *(output + 1 * candidates + i) - dh - h / 2;

            float x1 = x0 + w;
            float y1 = y0 + h;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);
            // 宽/高为0的直接舍弃，以免后续取整后rect roi为空矩阵
            if (x1 - x0 < 1 || y1 - y0 < 1)
                continue;
            float *mask_conf = new float[mask_number];
            for (size_t j = 0; j < mask_number; ++j)
            {
                mask_conf[j] = *(output + (5 + j) * candidates + i);
            }

            cv::Mat mask_conf_mat = cv::Mat(1, mask_number, CV_32F, mask_conf);
            mask_confs.push_back(mask_conf_mat);
            // labels.push_back(label);
            scores.push_back(score);
            bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, _score_thres, _iou_thres, indices);
    cv::Mat masks;
    int cnt = 0;
    for (auto &i : indices)
    {
        if (cnt >= _topk)
        {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        // obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }
    if (masks.empty())
    {
        // masks is empty
    }
    else
    {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {mask_w, mask_h});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * mask_w;
        int scale_dh = dh / input_h * mask_h;

        cv::Rect roi(scale_dw, scale_dh, mask_w - 2 * scale_dw,
                     mask_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height),
                       cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > _mask_conf;
        }
    }
}

/**
 * @brief 可视化
 * @param image
 * @param res
 * @param objs
 */
void FastSam::draw_objects(const Mat &image, Mat &res, const vector<Object> &objs)
{
    res = image.clone();
    Mat color_mask = image.clone();
    std::srand(std::time(0));
    for (auto &obj : objs)
    {
        // 生成随机的RGB值
        int r = std::rand() % 256;
        int g = std::rand() % 256;
        int b = std::rand() % 256;
        // if (obj.rect.area() < 20000)
        //     continue;
        cv::Scalar mask_color = {r, g, b};
        color_mask(obj.rect).setTo(mask_color, obj.boxMask);
    }
    cv::addWeighted(res, 0.5, color_mask, 0.8, 1, res);
    for (auto &obj : objs)
    {   // 过滤小目标框
        if (obj.rect.area() < 20000)
            continue;
        cv::rectangle(res, obj.rect, cv::Scalar(0, 0, 255), 2);
    }
}

FastSam::FastSam(const string &engine_file_path,const int warm_cnt) : Inference(engine_file_path)
{
    _input_size = cv::Size(1024,1024);
    _score_thres = 0.5;
    _iou_thres = 0.4;
    _mask_conf = 0.5f;
    _topk = 100;
    this->make_pipe(warm_cnt,true);
}

void FastSam::run(const Mat &frame, vector<Object> &objs)
{
    this->copy_from_Mat(frame, _input_size);
    this->infer();
    this->postprocess(objs);
}
