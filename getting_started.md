# Getting Started

[여기](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)에 설명되어 있는 대로 Jetson 디바이스에서 Tensorflow 설치를 진행하고, 아래 명령어를 입력하여 Protobuf 3.8.0을 설치합니다.

```shell
$ mkdir ${HOME}/project
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/jetson_nano.git
$ cd jetson_nano
$ ./install_protobuf-3.8.0.sh
$ source ${HOME}/.bashrc
```



다음으로, 아래 명령어를 입력하여 depencency 설치 및 TensorRT Engine을 빌드해줍니다.

```Shell
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/tensorrt_demos.git
$ cd ${HOME}/project/tensorrt_demos/ssd
$ ./install_pycuda.sh
$ sudo pip3 install onnx==1.4.1
$ cd ${HOME}/project/tensorrt_demos/plugins
$ make
```



위 과정이 모두 완료되었다면, 아래 명령어를 입력하여 yolo 모델을 다운받아주고, yolo_to_onnx.py와 onnx_to_tensorrt.py 코드를 실행시켜 TensorRT가 정상적으로 동작하는지 테스트 해줍니다.

```shell
$ cd ${HOME}/project/tensorrt_demos/yolo
$ ./download_yolo.sh
$ python3 yolo_to_onnx.py -m yolov3-416
$ python3 onnx_to_tensorrt.py -v -m yolov3-416
$ python3 yolo_to_onnx.py -m yolov4-416
$ python3 onnx_to_tensorrt.py -v -m yolov4-416
```



---

위 과정이 모두 실행되었다면, modelsearch-runtime repository를 clone 하여 가져온 다음 실행시켜줍니다.

```shell
$ git clone https://github.com/nota-github/modelsearch-runtime.git
$ pip3 install -r requirements.txt
```



아래 링크로 들어가서 runtime.py를 돌리기 위한 예시 모델과 이미지들을 다운받아줍니다.(Optional)

- https://drive.google.com/file/d/1Nu0xVcE-WRIDc-6wzSN-92bEPYH_v4Y1/view?usp=sharing



예시 명령어는 아래와 같습니다.

```shell
$ python3 ~/runtime.py --model ./30vyt516.trt --image_folder ./image/ --classes ./30vyt516_class.yaml
```

- --model : .trt 혹은 .tflite 확장자로 되어있는 모델 file 경로
- --image_folder : 최소 한 장 이상의 이미지가 들어있는 directory 경로
- --classes : 각 class 정보들이 담겨있는 yaml file 경로