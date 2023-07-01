# Using ETB (Edge Test Bed) for converting onnx models to trt models.

from enum import Enum
import traceback
import hashlib
import time, os

from etb.etb import etb_apis

_default_onnx_output_path = './temp/onnx/output.onnx'
_default_trt_output_path = './temp/tensorrt/output.trt'

class JetsonDevice(Enum):
    NANO='NANO'
    TX2='TX2'
    AGX_XAVIER='AGX_XAVIER'
    XAVIER_NX='XAVIER_NX'
    AGX_ORIN='AGX_ORIN'

def _build(model_file_path, model_file_name, additional_options, key):
    image_name = "nvcr.io/nvidia/tensorrt:23.02-py3"
    trtexec = '/usr/src/tensorrt/bin/trtexec'
    command =f'{trtexec} --onnx={model_file_name} --saveEngine=output_model.trt {additional_options} --buildOnly {key}'
    etb_apis.build(image_name, model_file_path, model_file_name, command, platform="linux/x86_64")

def run(jetson_type, onnx_path, output_file_path, additional_options):
    try:
        # nvidia: linux/arm64
        key = hashlib.md5(bytes(str(time.time()), 'utf-8')).hexdigest()
        head, tail = os.path.split(output_file_path)
        _build(onnx_path, tail, additional_options, key)
        # TODO: set node with given jetson_type.
        tid = etb_apis.run(nodes=['n1'])
        reports = etb_apis.wait_result(key)
        etb_apis.download(tid, filename=output_file_path)
        print(reports)

    except Exception as ex:
        traceback.print_exc()
        print(ex)

def export_onnx2trt(jetson_type=JetsonDevice.AGX_ORIN,
                    onnx_path=_default_onnx_output_path,
                    additional_options=["--fp16"],
                    output_path=_default_trt_output_path):
    run(jetson_type, onnx_path, output_path, additional_options)
    return
