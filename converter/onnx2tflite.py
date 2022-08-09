from enum import Enum
import os, sys, pathlib, subprocess, shutil
from utils.general import file_size
_default_openvino_output_path = './temp/openvino_output'
_default_openvino_xml_file_name = 'output.xml'
_default_openvino_bin_file_name = 'output.bin'
_default_openvino_map_file_name = 'output.mapping'
_default_tensorflow_file_path = './temp/tensorflow_output/output.h5'

class ModelDataType(Enum):
    FP32='FP32'
    FP16='FP16'
    INT8='INT8'

def export_onnx2openvino(onnx_path,
                        output_dir=_default_openvino_output_path,
                        dtype=ModelDataType.FP32):
    """
       Converting onnx to openvino
       Returns openvino output path and an exception if an exception is raised.
    """
    try:
        #check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        print(f'{__file__} starting export with openvino {ie.__version__}...')

        #os.makedirs('../opt/output', exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        cmd = f"mo --input_model {onnx_path} --output_dir {output_dir} --data_type {dtype.name}"
        subprocess.check_output(cmd.split())  # export

        print(f'{__file__} export success, saved as {output_dir} ({file_size(output_dir):.1f} MB)')
        file_list = os.listdir(output_dir)
        for ff in file_list:
            if pathlib.Path(ff).suffix == ".xml":
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_xml_file_name))
            elif pathlib.Path(ff).suffix == ".bin":
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_bin_file_name))
            else: # mapping
                tmp_path = os.path.join(output_dir, ff)
                os.rename(tmp_path, os.path.join(output_dir, _default_openvino_map_file_name))
        return output_dir, None
    except Exception as e:
        print(f'export failure: {e}')
        return output_dir, e
        

def export_openvino2tensorflow(openvino_path=_default_openvino_output_path,
                                openvino_xml_file_name=_default_openvino_xml_file_name,
                                tensorflow_file_path=_default_tensorflow_file_path):

    try:
        openvino_xml_file = os.path.join(openvino_path, openvino_xml_file_name)
        cmd = f"openvino2tensorflow --model_path {openvino_xml_file} --model_output_path {tensorflow_file_path} --output_saved_model"
        subprocess.check_output(cmd.split())  # export
        print(f'{__file__} export success, saved as {tensorflow_file_path} ({file_size(tensorflow_file_path):.1f} MB)')
        return tensorflow_file_path, None
    except Exception as e:
        print(f'export failure: {e}')
        return tensorflow_file_path, e


def export_tensorflow2tflite(dtype:str, output_dir, tensorflow_file_path=_default_tensorflow_file_path):
    # TF-Lite export
    try:
        print(f'{__file__} starting export with tflite...')

        output_dir = '../opt/output/'
        weights = "../opt/output/output_saved_model"
        if dtype == "FP32":
            tflite_model = convert_tflite_fp32(weights)    
            file_name = "output_fp32.tflite"
        elif dtype == "FP16":
            tflite_model = convert_tflite_fp32(weights)
            file_name = "output_fp16.tflite"
        else:
            tflite_model = convert_tflite_int8(weights)
            file_name = "output_int8.tflite"

        output_model_path = os.path.join(output_dir, file_name)
        with open(output_model_path, "wb") as f:
            f.write(tflite_model)                
        # remove openvino & tensorflow model 
        shutil.rmtree("../opt/output/output_openvino")#, ignore_errors=False, onerror=None)
        shutil.rmtree("../opt/output/output_saved_model")#, ignore_errors=False, onerror=None)

        print(f'{__file__} export success, saved as {output_model_path} ({file_size(output_model_path):.1f} MB)')
        return output_model_path
    except Exception as e:
        print(f'export failure: {e}')


def representative_dataset_gen():
    import numpy as np
    for i in range(0, np.shape(input)[0]):
        yield [np.array([input[i,:,:,:]], dtype=np.float32)]


def convert_tflite_fp32(model_path:str):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()        
    return tflite_model


def convert_tflite_fp16(model_path:str):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    return tflite_model


def convert_tflite_int8(model_path:str):
    import tensorflow as tf
    #integer only: int8_full.tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    converter.experimental_new_quantizer = True
    tflite_model = converter.convert()
    return tflite_model