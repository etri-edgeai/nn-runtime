# Quick Start



## Step 1

To install TensorFlow on Nvidia Jetson devices, please refer to [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). Protobuf 3.8.0 also has to be installed as follows:

```shell
$ mkdir ${HOME}/project
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/jetson_nano.git
$ cd jetson_nano
$ ./install_protobuf-3.8.0.sh
$ source ${HOME}/.bashrc
```





## Step 2

To install dependencies and build the TensorRT engine, please enter the commands as follow.

```Shell
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/tensorrt_demos.git
$ cd ${HOME}/project/tensorrt_demos/ssd
$ ./install_pycuda.sh
$ sudo pip3 install onnx==1.4.1
$ cd ${HOME}/project/tensorrt_demos/plugins
$ make
```





## Step 3

In order to check whether TensorRT works, you need to download a Yolo model, then sequentially execute ‘'yolo_to_onnx.py' and 'onnx_to_tensorrt.py' as follows:

```shell
$ cd ${HOME}/project/tensorrt_demos/yolo
$ ./download_yolo.sh
$ python3 yolo_to_onnx.py -m yolov3-416
$ python3 onnx_to_tensorrt.py -v -m yolov3-416
$ python3 yolo_to_onnx.py -m yolov4-416
$ python3 onnx_to_tensorrt.py -v -m yolov4-416
```





## Step 4

To use this runtime module, you need to clone this repository in your local storage and install dependencies that are described in‘'requirements.txt'.

```shell
$ git clone https://github.com/etri-edgeai/nn-runtime.git
$ pip3 install -r requirements.txt
```



To execute the 'runtime.py', you can use an example model with sample images that we provide, which can be accessed on this link below.

- [TBD]


Example command:

```shell
$ python3 ~/main.py --onnx=saved_resfpn34.onnx --target_framework=tflite
```

- --onnx : a path where .onnx file is located
- --model_py : a path where images are located
- --target_framework : tflite or trt
