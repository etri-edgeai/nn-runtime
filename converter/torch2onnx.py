from typing import Tuple

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
    except Exception as e:
        print(f'export to onnx error: {str(e)}')
        return False, e

    return True, None
