import argparse, os
from trt_converter import convert_all
from profiler import profile
from nn_runtime_checker import check_nn_runtime
from nn_runtime_code_generator import generatePythonFile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create NN-Runtime Infernece Code and Check the result")
    parser.add_argument('--onnx', type=str, help="The path to the onnx file")
    parser.add_argument('--input_width', type=int, help="The input width for the model")
    parser.add_argument('--input_height', type=int, help="The input height for the model")
    parser.add_argument('--output_folder', type=str, default="./output", help="The path to the onnx file")
    
    args = parser.parse_args()
    
    file_name, converted_file_infos = convert_all(file_path=args.onnx, input_width=args.input_width, input_height=args.input_height, output_folder=args.output_folder)
    
    info_dict = profile(converted_file_infos, os.path.join(args.output_folder, f"{file_name}.json"))
    
    check_nn_runtime(info_dict, os.path.join(args.output_folder, f"{file_name}.json"))
    
    generatePythonFile(info_dict, file_name, args.output_folder)
    
    