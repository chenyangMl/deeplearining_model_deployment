# 模型量化压缩和部署

深度学习任务的生产阶段需要根据实际的应用场景(硬件,交互方式)和功能需求(QPS,　速度，精度)，对模型做工程优化(量化，剪枝，蒸馏,,,), 再根据场景需求进行部署．



本项目仅以主体检测模型为例，汇总分享常用的工程部署部署方式，以供参考学习



模型训练参考工程: https://github.com/XinzeLee/PolygonObjectDetection



模型网络结构：　使用6.0以上的[yolov5n](mainareaDet/models/polygon_yolov5n.yaml)版本． 

| 模型名称           | 模型大小               | Imgsize       | 推理速度　(1080Ti) | 模型地址                                              |
| ------------------ | ---------------------- | ------------- | ------------------ | ----------------------------------------------------- |
| best.pth           | 4.6M(pf32), 2.5M(fp16) | (1,3,224,224) | 12ms               | [[download](mainareaDet/models/polygon_yolov5n.yaml)] |
| best.onnx          | 4.6M (FP32)            | /             | /                  |                                                       |
| best.engine        | 8.6M                   | /             |                    |                                                       |
| Ncnn (coming soon) | fp32 \| fp16 \| int8   |               |                    |                                                       |



## 模型推理测试

```
git clone https://github.com/chenyangMl/deeplearining_model_deployment.git
cd deeplearining_model_deployment

# 下载模型到 weights目录

单图推理(默认pth模型):  python mainareaDet/main.py
批量推理(默认pth模型):  python mainareaDet/main.py --bi 1
批量推理(使用engine模型):  python mainareaDet/main.py --bi 1 --mt engine
```



## 模型转换



```
转化成onnx模型: python export.py --export onnx
转化成engine模型(tensorrt): python export.py --export engine --device 0
转化成engine模型(tensorrt, 动态batch): python export.py --export engine --bs 32 --device 0 --dynamic
```



## 服务端部署

服务端的部署：就是将训练好的模型通过工具发布到物理服务器或云端服务器．工具指像TorchServe 或Triton这类的可以将模型以微服务的形式进行生命周期管理．通常工具

- 推理API     : REST and gRPC 均支持，
- 模型管理API: 　支持模型注册，卸载，查看.
- 指标监控API:  　支持系统级别的指标输出（系统指标和自定义指标）
- 进程管理，参数设置(batch_size)...
- 横向扩展便捷，横向扩展只需要扩展GPU服务即可．



基本流程：　训练好的模型A (demo.pth) -> 序列化模型B (demo.pt[torch] | demo.onnx[onnx] | demo.trt [tensorrt],...  )　-> 　工具接口封装模型B  -> 　部署



### Torchserve

TorchServe is a performant, flexible and easy to use tool for serving PyTorch models in production.　［[官网](https://pytorch.org/serve/)］

TorchServe官网定义是一个在生产环境高效，灵活和简易使用的PyTorch模型部署工具. 但亦能扩展支持别的深度学习框架，毕竟现在模型都是可以转换的.



torch2trt

1 环境配置: 参见[torch2trt官档](https://github.com/NVIDIA-AI-IOT/torch2trt)

```
安装torch2trt　　(示例版本torch2trt==0.3.0)
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install

安装TensorRT (示例版本tensorrt-7.2.2.3)
```



2：pycuda , TensorRT python -API

```
安装pycuda #参考 https://wiki.tiker.net/PyCuda/Installation/Linux/
git clone https://github.com/inducer/pycuda.git
$ cd pycuda-VERSION # if you're not there already
$ python configure.py --cuda-root=/where/ever/you/installed/cuda
$ su -c "make install"
```



TensorRT-python API

官方文档： https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

```
cd TensorRT-7.2.2.3/python
pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl  #根据自己python版本选择对应的whl
```





### Triton(TensorRT) Inference Server

...





## 移动端部署

### NCNN(Android)    | cpu







### 瑞芯微开发版　3586               |  NPU







