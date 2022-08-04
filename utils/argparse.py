import argparse

class ArgumentParserError(Exception): pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

def torch2onnx_argument_parser() -> ThrowingArgumentParser:
    parser = ThrowingArgumentParser()
    parser.add_argument('--torch',type=str, required=True, help='input torch model path (ex: /path/to/torch/file.pt')
    parser.add_argument('--onnx',type=str, required=True, help='output onnx model path (ex: /path/to/file.onnx')
    parser.add_argument('--input-shape', nargs='+', type=int, required=True, help='model input shape (ex: 1 3 128 128)')
    parser.add_argument('--input-names', nargs='+', default=['input'], help='input layer names (ex: input1 input2 input3)')
    parser.add_argument('--output-names', nargs='+', default=['output'], help='output layer names (ex: output1 output2 output3)')
    parser.add_argument('--opset-version',type=int, default=12, help='onnx operation set version (ex: 12')
    return parser

