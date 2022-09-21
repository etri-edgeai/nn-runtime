from pathlib import Path
from etb import etb_apis
import logging

def joinURL(*parts):
    return '/'.join([p.strip().strip('/') for p in parts])

class ETBClient():
    def __init__(self, etb_url: str = None) -> None:
        self.endpoint = etb_url
        self.nodes = etb_apis.get_nodes()

    @property
    def upload_model_url(self):
        return joinURL(self.endpoint, 'upload_model')

    def requestConvert(self, onnx_model_path: str, target_node, output_model_path: str, options=None):
        try:
            nodes = etb_apis.get_nodes()
            file = Path(onnx_model_path)
            with file.open() as f:
                logging.info(f'building trt from "{onnx_model_path}" on ETB....')
                etb_apis.build(self.upload_model_url, onnx_model_path, arguments=options)
                etb_apis.run(target_node, timeout=600)
                ret_json = etb_apis.wait_results()
                if ret_json is None:
                    raise Exception(f'failed to create trt model from {onnx_model_path} on ETB')

                #TODO: parse benchmark result.
                #TODO: save converted trt into output_model_path and return.
                result_json = {}
                return  result_json
        except Exception as e:
            logging.info(f'converting {onnx_model_path} to TensorRT on ETB is failed: {str(e)}')
            return None