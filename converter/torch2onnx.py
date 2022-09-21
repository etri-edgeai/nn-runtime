from typing import Tuple
import logging

def convert_torch2onnx(
    torch_model_path,
    output_onnx_model_path,
    input_shape: Tuple,
    input_names=['input'],
    output_names=['output'],
    opset_version=12):

    try:
        import torch
        torch_model = torch.load(torch_model_path)
        torch_model.eval()
        dummy_input = torch.rand(input_shape)
        torch.onnx.export(
            torch_model,
            dummy_input,
            output_onnx_model_path,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version)
        import onnx
        model_onnx = onnx.load(output_onnx_model_path)  # load onnx model
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), output_onnx_model_path)
    except Exception as e:
        logging.info(f'export to onnx error: {str(e)}')
        return False, e

    return True, None

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.argparse import parse_torch2onnx_arguments
    values = parse_torch2onnx_arguments(sys.argv)
    convert_torch2onnx(
        torch_model_path=values.get('torch_model_path'),
        output_onnx_model_path=values.get('output_onnx_model_path'),
        input_shape=values.get('input_shape'),
        input_names=values.get('input_names'),
        output_names=values.get('output_names'),
        opset_version=values.get('opset_version')
    )    