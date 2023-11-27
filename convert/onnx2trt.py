import tensorrt as trt
import os
import argparse
from inference_trt import FastSam
import cv2
import time

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger() # 默认trt.Logger.WARINING
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  #  使用trt.Logger.VERBOSE时将会有大量输出，不建议开启 ，此外还有trt.Logger.ERROR 

def get_engine(onnx_file_path, engine_file_path="",replace=False,FP_16=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # config.max_workspace_size = 1 << 32  # tensorRT 7.x r写法
             # 最大内存占用
            # 显存溢出需要重新设置
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 4GB tensorRT 8.x r写法
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found !".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # # 固定动态输入onnx模型尺寸
            # network.get_input(0).shape=[1,3,640,640]
            # 配置动态输入onnx模型优化
            profile = builder.create_optimization_profile()
            profile.set_shape("input.1",(1, 3, 1024, 1024), (1, 3, 1024, 1024),(1, 4, 1024, 1024))
            config.add_optimization_profile(profile)
            if builder.platform_has_fast_fp16 and FP_16:
                config.set_flag(trt.BuilderFlag.FP16)
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine
    # 覆盖写入engine文件(如果此前存在)
    if replace:
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print("Cannot remove existing file: ",engine_file_path)
    # 如果存在则直接读取，否则对onnx文件进行转换后再返回
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='models/FastSAM-s.onnx', help='ONNX weights path')
    parser.add_argument('--engine', type=str, default='models/FastSAM-s.engine', help='output engine model path')
    parser.add_argument('--image', type=str, default="images/cat.jpg", help='test image path')
    opt = parser.parse_args()
    onnx_file_path = opt.onnx
    engine_file_path = opt.engine
    test_image_path = opt.image
    get_engine(onnx_file_path, engine_file_path)
    model = FastSam(model_weights=engine_file_path)
    img = cv2.imread(test_image_path)
    start = time.time()
    masks = model.segment(img)
    print(f'cost time {time.time()-start} !')
    print("[Ouput]: ", masks.shape)


if __name__ == "__main__":
    main()
