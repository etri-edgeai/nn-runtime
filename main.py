from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model_loader import load_model
from resnet_fpn import get_pose_net
from utils.argparse import ModelDataType
import torch
import argparse
import os

from converter.onnx2tflite import export_tensorflow2tflite, export_onnx2tensorflow
from converter.onnx2jetsontrt import export_onnx2trt, JetsonDevice

from packager.builder import build_package, Platform, Framework, PythonVersion

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@torch.no_grad()
def export_onnx(opt):
    

    model = get_pose_net(num_layers=opt.num_layers, head_conv=opt.head_conv, heads=opt.heads)
    model = load_model(model, opt.weight_path)
    # torch.save(model, pt_file_path)
    #model.eval()
    dynamic_axes = {'input' : {0 : 'batch_size'},
                    'output_0' : {0 : 'batch_size'},
                    'output_1' : {0 : 'batch_size'},
                    'output_2' : {0 : 'batch_size'},
                    'output_3' : {0 : 'batch_size'}}
    torch.onnx.export(model, torch.rand(1, 3, 576, 1024),
                    opt.onnx, input_names=['input'],
                    output_names=['output_0','output_1','output_2','output_3'],
                    opset_version=13)
    
def build_runtime(onnx_file_path = 'saved_resfpn34.onnx', model_py_path = ''):

    tensorflow_dir = './temp/dist/tensorflow'
    tflite_dir = 'dist/tflite'
    export_onnx2tensorflow(onnx_path=onnx_file_path, output_dir=tensorflow_dir)
    tflite_model_path = export_tensorflow2tflite(dtype=ModelDataType.FP16, tensorflow_path=tensorflow_dir, output_dir=tflite_dir)
    tflite_package_data = {
        "model_url": tflite_model_path,
        "source_path": model_py_path,
        "package_name": "resfpn34",
        "platform": Platform.linux_x64,
        "framework": Framework.tflite,
        "package_version": "0.0.1",
        "obf": False,
        "python_version": PythonVersion.py39
    }
    build_package(data=tflite_package_data, output_dir="dist/package")

    # trt_file_path = 'dist/ouput.trt'
    # export_onnx2trt(jetson_type=JetsonDevice.AGX_ORIN, onnx_path=onnx_file_path, additional_options="--fp16", output_path=trt_file_path)
    # trt_package_data = {
    #     "model_url": trt_file_path,
    #     "source_path": model_py_path,
    #     "package_name": "resfpn34",
    #     "platform": Platform.linux_x64,
    #     "framework": Framework.trt,
    #     "package_version": "0.0.1",
    #     "obf": False,
    #     "python_version": PythonVersion.py39
    # }
    # build_package(data=trt_package_data, output_dir="dist/package")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=dict, default={'hm': 1, 'wh': 2, 'id':128, 'reg': 2}, help="network head output channels")
    parser.add_argument("--head_conv", type=int, default=128, help="network head output channels")
    parser.add_argument("--num_layers", type=int, default=34, help="ResNet-fpn layers")
    parser.add_argument("--weight_path", type=str, default='./resfpn34_weights.pth', help="pretrained weights path")
    parser.add_argument("--onnx", type=str, default='./saved_resfpn34.onnx', help="onnx file path")
    parser.add_argument("--model_py", type=str, default='./resfpn34_model.py', help="pre/post processor python file path")
    opt = parser.parse_args()
    try:
        export_onnx(opt)
    except Exception as e:
        print(e)
        pass

    build_runtime(onnx_file_path = opt.onnx, model_py_path = opt.model_py)
