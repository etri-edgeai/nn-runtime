# Using ETB (Edge Test Bed) for converting onnx models to trt models.

from enum import Enum
from utils.argparse import ModelDataType
from etb.client import ETBClient

_default_onnx_output_path = './temp/onnx/output.onnx'
_default_trt_output_path = './temp/tensorrt/output.trt'

class JetsonDevice(Enum):
    NANO='NANO'
    TX2='TX2'
    AGX_XAVIER='AGX_XAVIER'
    XAVIER_NX='XAVIER_NX'
    AGX_ORIN='AGX_ORIN'

def export_onnx2trt(dtype=ModelDataType.FP32,
                    jetson_type=JetsonDevice.XAVIER_NX,
                    onnx_path=_default_onnx_output_path,
                    additional_options=["--FP16"],
                    output_dir=_default_trt_output_path):
    client = ETBClient("put etb url here")
    #TODO: match device
    target_node = [node for node in client.nodes if node.name.lower() == jetson_type.value.lower()]
    client.requestConvert(onnx_model_path=onnx_path,
                        target_node=target_node,
                        output_model_path=output_dir,
                        options=additional_options)
    return
