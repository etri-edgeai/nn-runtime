import argparse
import sys
from enum import Enum

class ModelDataType(Enum):
    FP32='FP32'
    FP16='FP16'

class ArgumentParserError(Exception): pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

def default_torch2_argument_parser() -> ThrowingArgumentParser:
    parser = ThrowingArgumentParser()
    parser.add_argument('--torch', type=str, required=True, help='input torch model path (ex: /path/to/torch/file.pt')
    parser.add_argument('--input-shape', nargs='+', type=int, required=True, help='model input shape (ex: 1 3 128 128)')
    parser.add_argument('--input-names', nargs='+', default=['input'], help='input layer names (ex: input1 input2 input3)')
    parser.add_argument('--output-names', nargs='+', default=['output'], help='output layer names (ex: output1 output2 output3)')
    parser.add_argument('--opset-version', type=int, default=12, help='onnx operation set version (ex: 12')
    return parser

def torch2onnx_argument_parser() -> ThrowingArgumentParser:
    parser = default_torch2_argument_parser()
    parser.add_argument('--onnx', type=str, required=True, help='output onnx model path (ex: /path/to/file.onnx')
    return parser

def parse_torch2onnx_arguments(input) -> dict:
    parser = torch2onnx_argument_parser()
    args, _ = parser.parse_known_args(input)
    #TODO: show warnings for unknown input arguments.

    return {
        'torch_model_path': args.torch,
        'output_onnx_model_path': args.onnx,
        'input_shape': tuple(args.input_shape),
        'input_names': args.input_names,
        'output_names': args.output_names,
        'opset_version': args.opset_version,
    }

def torch2tflite_argument_parser() -> ThrowingArgumentParser:
    parser = default_torch2_argument_parser()
    parser.add_argument('--tflite', type=str, required=True, help='output onnx model path (ex: /path/to/file.tflite')
    parser.add_argument('--dtype', type=ModelDataType, default=ModelDataType.FP32, choices=list(ModelDataType), help='model data type (available: FP32, FP16)')
    return parser

def parse_torch2tflite_arguments(input=sys.argv[1:]) -> dict:
    parser = torch2tflite_argument_parser()
    args, _ = parser.parse_known_args(input)
    #TODO: show warnings for unknown input arguments.

    return {
        'torch_model_path': args.torch,
        'output_tflite_model_path': args.tflite,
        'input_shape': tuple(args.input_shape),
        'input_names': args.input_names,
        'output_names': args.output_names,
        'opset_version': args.opset_version,
        'dtype': args.dtype,
    }