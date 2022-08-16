# dev_NVIDIA-Docker
Holding Place for NVIDIA development and files

#### CUDA Programming Links
[https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/) <br/>

[https://developer.nvidia.com/blog/even-easier-introduction-cuda/](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) <br/>

[https://developer.nvidia.com/blog/?s=parallel+for+all](https://developer.nvidia.com/blog/?s=parallel+for+all) <br/>

[https://developer.nvidia.com/blog/cutting-edge-parallel-algorithms-research-cuda/](https://developer.nvidia.com/blog/cutting-edge-parallel-algorithms-research-cuda/
) <br/>

[https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/](https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/) <br/>

[https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/
) <br/>

[https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/](https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/) <br/>

[https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/) <br/>

[https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/](https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/) <br/>

NVIDIA CUDA development and encapsulation using Docker<br/>
![NVIDIA Docker](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker.png)

#### NVIDIA Container Link
[https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda](
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) <br/>

#### NVIDIA Container Runtime Repository
[https://nvidia.github.io/nvidia-container-runtime/](https://nvidia.github.io/nvidia-container-runtime/) <br/>

#### NVIDIA Toolkit Instructions
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) <br/>

#### NVIDIA Tesla Cards
- M2090 ***CUDA 8 is the last to support this architecture***
- M40
- K40
- K80

#### CUDA, cuDNN
- GPU Accelerated libraries [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download) <br/>
- [Install PyCUDA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda)

#### Tensorflow GPU Support
- [Tensorflow Docker](https://www.tensorflow.org/install/docker)
- [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)
- [Install with PIP](https://www.tensorflow.org/install/pip)
- [Pypi Tensorflow-gpu](https://pypi.org/project/tensorflow-gpu/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

#### Benchmarks
- AI-Benchmark [https://ai-benchmark.com/alpha.html](https://ai-benchmark.com/alpha.html) <br/>
- Lambda Tensorflow [https://github.com/lambdal/lambda-tensorflow-benchmark](https://github.com/lambdal/lambda-tensorflow-benchmark)

#### Links
[NVIDIA/Docker Github](https://github.com/NVIDIA/nvidia-docker) <br/>
[NVIDIA Catalog https://ngc.nvidia.com/catalog/containers/nvidia:pytorch](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) <br/>
[Individual NVIDIA/Docker PyTorch](https://github.com/anibali/docker-pytorch) <br/>


#### Step-by-Step Configuration Guide
1) Install GPU
2) Install GPU Driver (440.64) and CUDA Toolkit (10.2)
- Verify Hardware and Software
```
$nvidia-smi
```
![nvidia-smi cmd](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-smi-02.png) <br/>

4) Install nvidia-docker2 and reload the Docker daemon configuration
```
$sudo apt-get install -y nvidia-docker2
$sudo pkill -SIGHUP dockerd
```
5) Test nvidia-smi with the latest official CUDA image that works with your GPU (10.2)
```
$docker run --runtime=nvidia --rm nvidia/cuda:10.2-base nvidia-smi
```
![nvidia-smit test](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-smi_testimage-01.png) <br/>

#### Get NVIDIA Toolkit and Cudnn8 Working on Ubuntu 19.10, Install and Run Tensorflow 2.2
Ran following python code (as tesorflow2_test.py):
```
import tensorflow as tf
print(tf.__version__)

tf.test.is_gpu_available(
    cuda_only = True,
    min_cuda_compute_capability=None
)

print(tf.test.is_gpu_available)

```

*** Ran into errors *** <br/>
Solution was to install cuda-10.0, cuda-10.1, cuda-10.2 with --override switch for compiler check.
Otherwise, script will have errors with libraries.

#### Cannot find Kernel Headers Error
Solution: <br/>
```
$sudo yum install "kernel-devel-uname-r == $(uname -r)"
```
