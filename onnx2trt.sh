# 优先建议使用trtexec工具进行onnx2trt，一般位于下载的TensorRT包的bin目录下
# 建议onnx以及saveEngine使用绝对路径
/home/xxx/TensorRT-8.4.1.5/bin/trtexec --onnx=/home/xxx/projects/fastsam/models/fast_sam_x.onnx \
                                --saveEngine=/home/xxx/projects/fastsam/models/fast_sam_x.engine \
                                --explicitBatch \
                                --minShapes=images:1x3x1024x1024 \
                                --optShapes=images:1x3x1024x1024 \
                                --maxShapes=images:4x3x1024x1024 \
                                --verbose \
                                --device=0

# 使用python脚本转换时需要本机CUDA版本与pytorch cuda版本严格一致
# 否则会报错 [TRT] [W] CUDA initialization failure with error: 35
# python3 convert/onnx2trt.py --onnx "models/fast_sam_x.onnx" \
#                     --engine "models/fast_sam_x.engine" \
#                     --image "images/cat.jpg"