#pragma once

#include <dirent.h>
#include <memory.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "common.hpp"
#include "NvInferPlugin.h"

using cv::Mat, cv::Size_;
using std::string, std::vector;

class Inference
{
public:
    /**
     * @brief Inference推理器基类
     * @param engine_file_path trt模型路径
     */
    explicit Inference(const std::string &engine_file_path)
    {
        std::ifstream file(engine_file_path, std::ios::binary);
        assert(file.good());
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        char *trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        model_name = engine_file_path.substr(engine_file_path.find_last_of('/') + 1, engine_file_path.length());
        initLibNvInferPlugins(&this->gLogger, "");
        this->runtime = nvinfer1::createInferRuntime(this->gLogger);
        assert(this->runtime != nullptr);

        this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
        assert(this->engine != nullptr);

        this->context = this->engine->createExecutionContext();

        assert(this->context != nullptr);
        cudaStreamCreate(&this->stream);
        this->num_bindings = this->engine->getNbBindings();

        for (int i = 0; i < this->num_bindings; ++i)
        {
            Binding binding;
            nvinfer1::Dims dims;
            nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
            std::string name = this->engine->getBindingName(i);
            binding.name = name;
            binding.dsize = type_to_size(dtype);

            bool IsInput = engine->bindingIsInput(i);
            if (IsInput)
            {
                this->num_inputs += 1;
                dims = this->engine->getProfileDimensions(
                    // nvinfer1::OptProfileSelector::kOPT,kMAX,kMIN 对应生成trt模型时的优化选项
                    i, 0, nvinfer1::OptProfileSelector::kOPT);
                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->input_bindings.push_back(binding);
                // set max opt shape
                this->context->setBindingDimensions(i, dims);
            }
            else
            {
                dims = this->context->getBindingDimensions(i);
                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->output_bindings.push_back(binding);
                this->num_outputs += 1;
            }
        }
    };

    /**
     * @brief 预分配输入输出内存显存空间,warmup
     * @param warmup
     */
    void make_pipe(int cnt, bool warmup = true)
    {
        for (auto &bindings : this->input_bindings)
        {
            void *d_ptr;
            CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize,
                                  this->stream));
            this->device_ptrs.push_back(d_ptr);
        }

        for (auto &bindings : this->output_bindings)
        {
            void *d_ptr, *h_ptr;
            size_t size = bindings.size * bindings.dsize;
            CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
            CHECK(cudaHostAlloc(&h_ptr, size, 0));
            this->device_ptrs.push_back(d_ptr);
            this->host_ptrs.push_back(h_ptr);
        }

        if (warmup)
        {
            for (int i = 0; i < cnt; i++)
            {
                for (auto &bindings : this->input_bindings)
                {
                    size_t size = bindings.size * bindings.dsize;
                    void *h_ptr = malloc(size);
                    memset(h_ptr, 0, size);
                    CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size,
                                          cudaMemcpyHostToDevice, this->stream));
                    free(h_ptr);
                }
                this->infer();
            }
            printf("model %s warmup %d times\n", model_name.c_str(), cnt);
        }
    };

    /**
     * @brief 析构,释放显存
     */
    ~Inference()
    {
        delete context;
        delete engine;
        delete runtime;
        cudaStreamDestroy(this->stream);
        for (auto &ptr : this->device_ptrs)
        {
            CHECK(cudaFree(ptr));
        }

        for (auto &ptr : this->host_ptrs)
        {
            CHECK(cudaFreeHost(ptr));
        }
    };

    /**
     * @brief trt推理api
     */
    void infer()
    {
        // DMA input batch data to device, infer on the batch asynchronously, and
        // DMA output back to host
        this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
        for (int i = 0; i < this->num_outputs; i++)
        {
            size_t osize =
                this->output_bindings[i].size * this->output_bindings[i].dsize;
            CHECK(cudaMemcpyAsync(this->host_ptrs[i],
                                  this->device_ptrs[i + this->num_inputs], osize,
                                  cudaMemcpyDeviceToHost, this->stream));
        }
        cudaStreamSynchronize(this->stream);
    };

protected:
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    string model_name;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

using inference_ptr = std::unique_ptr<Inference>;