import os, subprocess, logging
from utils.general import file_size
from utils.argparse import ModelDataType

_default_tensorflow_output_path = './temp/tensorflow_output'
_default_tflite_output_path = './temp/tflite_output'

def export_onnx2tensorflow(onnx_path, output_dir=_default_tensorflow_output_path):
    """
       Converting onnx to tensorflow
       Returns exported tensorflow output path and an exception if an exception is raised.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        cmd = f"onnx-tf convert -i {onnx_path} --outdir {output_dir}"
        subprocess.check_output(cmd.split())  # export
        logging.info(f'{__file__} export success, saved as {output_dir} ({file_size(output_dir):.1f} MB)')
        return output_dir, None
    except Exception as e:
        logging.info(f'export failure: {e}')
        return output_dir, e

def export_tensorflow2tflite(dtype=ModelDataType.FP32,
                            tensorflow_path=_default_tensorflow_output_path,
                            output_dir=_default_tflite_output_path):
    # TF-Lite export
    os.makedirs(output_dir, exist_ok=True)
    try:
        logging.info(f'{__file__} starting export with tflite...')
        if dtype.name == "FP16":
            tflite_model = convert_tflite_fp16(tensorflow_path)
            file_name = "output_fp16.tflite"
        else:
            tflite_model = convert_tflite_fp32(tensorflow_path)    
            file_name = "output_fp32.tflite"

        output_model_path = os.path.join(output_dir, file_name)
        with open(output_model_path, "wb") as f:
            f.write(tflite_model)                

        logging.info(f'{__file__} export success, saved as {output_model_path} ({file_size(output_model_path):.1f} MB)')
        return output_model_path

    except Exception as e:
        logging.info(f'export failure: {e}')

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
