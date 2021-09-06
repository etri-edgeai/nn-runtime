# Runtime

아래 링크로 들어가서 runtime.py를 돌리기 위한 예시 모델과 이미지들을 다운받아줍니다.

- https://drive.google.com/file/d/1Nu0xVcE-WRIDc-6wzSN-92bEPYH_v4Y1/view?usp=sharing

  

예시 명령어는 아래와 같습니다.

1. trt

```shell
python3 ~/runtime.py --model ./30vyt516.trt --image_folder ./image/ --classes ./30vyt516_class.yaml
```

2. tflite

```shell
python3 ~/runtime.py --model ./30vyt516.tflite --image_folder ./image/ --classes ./30vyt516_class.yaml
```

