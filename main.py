from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model_loader import load_model
from resnet_fpn import get_pose_net
from utils.argparse import ModelDataType
import torch
import argparse
import csv

from packager.builder import build_package, Platform, Framework, PythonVersion

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def analyze_profile_data(model_class_names, csv_profile_paths, trt_file_paths, trt_file_names):
    engine_class_names = []
    engine_file_paths = []
    engine_file_names = []

    best_indexs = [0, 0, 0, 0]
    best_counts = [0, 0, 0, 0]
    
    for index, csv_file in enumerate(csv_profile_paths):
        inference_count = {0:0, 1:0, 2:0, 3:0,}
        with open('names.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row['first_name'], row['last_name'])
                cpu_usage = float(row[' cpu_usage - %'])
                mem_usage = float(row[' memory_usage - %'])
            if cpu_usage >=0 and cpu_usage < 75 and mem_usage >= 0 and mem_usage < 75:
                inference_count[0] = inference_count[0] + 1 
            elif cpu_usage >=75 and mem_usage >= 0 and mem_usage < 75:
                inference_count[1] = inference_count[1] + 1
            elif cpu_usage >=0 and cpu_usage < 75 and mem_usage >= 75:
                inference_count[2] = inference_count[2] + 1
            elif cpu_usage >=75 and mem_usage >= 75:
                inference_count[3] = inference_count[3] + 1
    return engine_class_names, engine_file_paths, engine_file_names

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
    
def build_runtime(onnx_file_path = 'saved_resfpn34.onnx',
                  model_py_path = "resfpn34_model.py",
                  target_framework = 'tflite',
                  package_name = 'resfpn34',
                  input_layer_names: [str] = None,
                  output_layer_names: [str] = None,
                  enable_dla: bool = False,
                  is_obf = False):
    inference_class_name = 'NPModel'
    if target_framework == 'tflite' :
        from converter.onnx2tflite import export_tensorflow2tflite, export_onnx2tensorflow
        tensorflow_dir = './temp/dist/tensorflow'
        tflite_dir = 'dist/tflite'
        export_onnx2tensorflow(onnx_path=onnx_file_path, output_dir=tensorflow_dir)
        tflite_model_path = export_tensorflow2tflite(dtype=ModelDataType.FP16, tensorflow_path=tensorflow_dir, output_dir=tflite_dir)
        tflite_package_data = {
            "model_url": tflite_model_path,
            "source_path": model_py_path,
            "package_name": package_name,
            "platform": Platform.linux_x64,
            "framework": Framework.tflite,
            "package_version": "0.0.1",
            "obf": is_obf,
            "python_version": PythonVersion.py39
        }
        build_package(data=tflite_package_data, output_dir="dist/package")
    else:
        from converter.onnx2jetsontrt import export_onnx2trt, JetsonDevice

        model_class_names = [f'{inference_class_name}FP16',
                       f'{inference_class_name}FP16Sparsity',
                       f'{inference_class_name}Int8',
                       f'{inference_class_name}Int8Sparsity',
                       f'{inference_class_name}Mixed',
                       f'{inference_class_name}MixedSparsity',]
        
        trt_file_paths = ['dist/ouput_fp16.trt',
                          'dist/ouput_fp16_sparsity.trt',
                          'dist/ouput_int8.trt',
                          'dist/ouput_int8_sparsity.trt',
                          'dist/ouput_mix.trt',
                          'dist/ouput_mix_sparsity.trt']
        
        trt_file_names = ['ouput_fp16.trt',
                          'ouput_fp16_sparsity.trt',
                          'ouput_int8.trt',
                          'ouput_int8_sparsity.trt',
                          'ouput_mix.trt',
                          'ouput_mix_sparsity.trt']
        
        csv_profile_paths = ['dist/ouput_fp16.csv',
                              'dist/ouput_fp16_sparsity.csv',
                              'dist/ouput_int8.csv',
                              'dist/ouput_int8_sparsity.csv',
                              'dist/ouput_mix.csv',
                              'dist/ouput_mix_sparsity.csv']
        trt_build_options = [
            '--fp16',
            '--fp16 --sparsity=enable',
            '--int8',
            '--int8 --sparsity=enable',
            '--best',
            '--best --sparsity=enable',
        ]

        if enable_dla:
            model_class_names.append(f'{inference_class_name}FP16SparsityDLA')
            model_class_names.append(f'{inference_class_name}Int8SparsityDLA')
            model_class_names.append(f'{inference_class_name}MixedSparsityDLA')
            trt_file_paths.append('dist/ouput_fp16_sparsity_dla.trt')
            trt_file_paths.append('dist/ouput_int8_sparsity_dla.trt')
            trt_file_paths.append('dist/ouput_mix_sparsity_dla.trt')
            trt_file_names.append('ouput_fp16_sparsity_dla.trt')
            trt_file_names.append('ouput_int8_sparsity_dla.trt')
            trt_file_names.append('ouput_mix_sparsity_dla.trt')
            csv_profile_paths.append('dist/ouput_fp16_sparsity_dla.csv')
            csv_profile_paths.append('dist/ouput_int8_sparsity_dla.csv')
            csv_profile_paths.append('dist/ouput_mix_sparsity_dla.csv')
            trt_build_options.append('--fp16 --sparsity=enable --useDLACore=0 --allowGPUFallback')
            trt_build_options.append('--int8 --sparsity=enable --useDLACore=0 --allowGPUFallback')
            trt_build_options.append('--best --sparsity=enable --useDLACore=0 --allowGPUFallback')

        for i, engine_path in enumerate(trt_file_paths):
            build_option_string = trt_build_options[i]
            export_onnx2trt(jetson_type=JetsonDevice.AGX_ORIN,
                            onnx_path=onnx_file_path,
                            additional_options=build_option_string,
                            output_path=engine_path,
                            profile_output_path=csv_profile_paths[i])
        #TODO: select 4 engines
        engine_class_names, engine_file_paths, engine_file_names = analyze_profile_data(model_class_names, csv_profile_paths, trt_file_names)

        trt_package_data = {
            "inference_class_name": inference_class_name,
            "model_class_names": engine_class_names,
            "model_urls": engine_file_paths,
            "eingine_file_names": engine_file_names,
            "source_path": model_py_path,
            "package_name": package_name,
            "platform": Platform.linux_x64,
            "framework": Framework.trt,
            "package_version": "0.0.1",
            "obf": is_obf,
            "python_version": PythonVersion.py39
        }
        build_package(data=trt_package_data, output_dir="dist/package")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=dict, default={'hm': 1, 'wh': 2, 'id':128, 'reg': 2}, help="network head output channels")
    parser.add_argument("--head_conv", type=int, default=128, help="network head output channels")
    parser.add_argument("--num_layers", type=int, default=34, help="ResNet-fpn layers")
    parser.add_argument("--weight_path", type=str, default='./resfpn34_weights.pth', help="pretrained weights path")
    parser.add_argument("--onnx", type=str, default='saved_resfpn34.onnx', help="onnx file path")
    parser.add_argument("--model_py", type=str, default='resfpn34_model.py', help="pre/post processor python file path")
    parser.add_argument("--target_framework", type=str, default='tflite', help="tflite or trt")
    parser.add_argument("--package_name", type=str, default='resfpn34', help="python package name which will be generated.")
    parser.add_argument("--obfuscate", type=bool, default=False, help="default : False,  If True, obfuscated python package will be generated.")
    parser.add_argument('--input_layer_names', nargs='*', help="name of input layers")
    parser.add_argument('--output_layer_names', nargs='*', help="name of output layers")
    parser.add_argument("--enable_dla", type=bool, default=False, help="default : False,  If True, it enables dla gpu fallback option on Jetson Devices if it's possible.")
    opt = parser.parse_args()
    try:
        export_onnx(opt)
    except Exception as e:
        print(e)
        pass

    build_runtime(onnx_file_path = opt.onnx,
                  model_py_path = opt.model_py,
                  target_framework = opt.target_framework,
                  package_name = opt.package_name,
                  is_obf = opt.obfuscate)
