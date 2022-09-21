import requests
from urllib.parse import urljoin
from pathlib import Path

def joinURL(*parts):
    return '/'.join([p.strip().strip('/') for p in parts])

class ETBClient():
    def __init__(self, etb_url: str = None) -> None:
        self.endpoint = etb_url

    @property
    def upload_model_url(self):
        return joinURL(self.endpoint, 'upload_model')

    def requestConvert(self, onnx_model_path: str, output_model_path: str, options=None):
        try:
            file = Path(onnx_model_path)
            with file.open() as f:
                print(f'upload file "{onnx_model_path}" to ETB....')
                upload_file = {'file':file}
                upload_result = requests.request(method='post', url=self.endpoint, files=upload_file)
                if upload_result.ok is not True:
                    raise Exception(f'failed to upload {onnx_model_path} to ETB')
                print(f'{onnx_model_path} is uploaded.')
                json_body = {}
                #TODO: parse result and build benchmark request json
                print(f'request benchmark with options {options}')
                benchmark_result = requests.request(method='post', url=self.endpoint, json=json_body)
                if benchmark_result.ok is not True:
                    raise Exception(f'benchmark failed: {benchmark_result.reason}')
                #TODO: parse benchmark result and return.
                result_json = {}
                return  result_json
        except Exception as e:
            print(f'convert {onnx_model_path} to TensorRT failed: {str(e)}')
            return None