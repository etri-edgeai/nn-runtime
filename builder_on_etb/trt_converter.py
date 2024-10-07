import argparse, os, onnx, subprocess
from np_utils import save_dict_to_json
data_types = [
        "fp32",
        "fp16",
        "fp16",
        "fp16",
        "int8",
        "int8",
        "int8",
        "best",
        "best",
        "best",
    ]
sparsity = [
    False,
    False,
    True,
    True,
    False,
    True,
    True,
    False,
    True,
    True,
]

use_dla = [
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
]

def get_input_layer_dict(model_file_path:str):
    """
    Return dict[input_layer_name, input_layer_shape]
    ex) For double input layer model, inputs = {'images': [3, 1024, 1024], 'input_noise': [3, 1024, 1024]}
    """
    model = onnx.load(model_file_path,load_external_data=False)
    inputs = {}
    outputs = {}
    for inp in model.graph.input:
        dim_values = []
        for dim in inp.type.tensor_type.shape.dim:
            if str(dim.dim_value).isdigit() and dim.dim_value > 0:
                dim_values.append(dim.dim_value)
            else:
                dim_values.append(-1)
        # shape = str(inp.type.tensor_type.shape.dim).replace("[","").replace("]","")
        inputs[inp.name] = dim_values
    for outp in model.graph.output:
        dim_values = []
        for dim in outp.type.tensor_type.shape.dim:
            if str(dim.dim_value).isdigit() and dim.dim_value > 0:
                dim_values.append(dim.dim_value)
            else:
                dim_values.append(-1)
        # shape = str(inp.type.tensor_type.shape.dim).replace("[","").replace("]","")
        outputs[outp.name] = dim_values
    return inputs, outputs

def check_input_demensions(input_layers_info):
    needs_batch_size = False
    needs_input_layer_dimensions = False
    #check input dimensions
    for _, key in enumerate(input_layers_info):
        value = input_layers_info[key]
        if len(value) > 4:
            #it's not CV Model cannot handle the input layer
            continue
        if value[0] < 0:
            needs_batch_size = True
        if value[1] < 0 or value[2] < 0 or value[3] < 0:
            needs_input_layer_dimensions = True

        if needs_batch_size or needs_input_layer_dimensions:
            break
    return needs_batch_size, needs_input_layer_dimensions

def convert(file_path,
            data_type="fp32",
            sparsity=False,
            use_dla=False,
            batch_size = 1, channel_size = 3, input_width = 512, input_height = 512,
            output_folder="./output"):
    
    input_shape = f"{batch_size}x{channel_size}x{input_height}x{input_width}"
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    trt_file_name = file_name
    command_line = f"/usr/src/tensorrt/bin/trtexec --onnx={file_path}"
    
    input_layers_info, output_layer_info = get_input_layer_dict(file_path)
    needs_batch_size, needs_input_layer_dimensions = check_input_demensions(input_layers_info)
    shape_string = ""
    for _, key in enumerate(input_layers_info):
        if len(shape_string) > 0:
            shape_string += ","    
        shape_string += f"{key}:{input_shape}"
    shape_option_string = f" --minShapes={shape_string} --maxShapes={shape_string}"

    if needs_batch_size or needs_input_layer_dimensions:
        command_line += shape_option_string
    
    
    if data_type == "best" :
        trt_file_name += "_mix"
    else:
        trt_file_name += f"_{data_type}"
        
    if data_type in ["fp16", "int8", "best"]:
        command_line += f" --{data_type}"
        
    if sparsity is True:
        command_line += f" --sparsity=enable"
        trt_file_name += f"_sparsity"
    
    if use_dla is True:
        command_line += f" --allowGPUFallback --useDLACore=0"
        trt_file_name += f"_dla"
    
    trt_file_name+=".trt"
    output_file_path = os.path.join(output_folder, trt_file_name)
    command_line += f" --saveEngine={output_file_path} > /dev/null 2>&1"
    print(f"Creating Engine: { os.path.abspath(output_file_path)}\n data type: {data_type}\n enable sparsity: {sparsity}\n use dla: {use_dla}")
    print(command_line)
    
    ##################
    ### CONVERSION ###
    stream = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, convert_stderr = stream.communicate()
    error_string = str(convert_stderr.decode('utf-8')).strip()
    if len(error_string) > 0:
        print(error_string)

    ##################
    result_dict = {
                "trt_file_name": trt_file_name,
                "file_path": os.path.abspath(output_file_path),
                "data_type": data_type,
                "sparsity": sparsity,
                "use_dla" : use_dla,
                "input_names": list(input_layers_info.keys()),
                "output_names": list(output_layer_info.keys()),
            }
    print(f"Creating Engine Done.")
    return file_name, result_dict
    
def convert_all(file_path, batch_size = 1, channel_size = 3, input_width = 512, input_height = 512, output_folder="./output"):
    
    converted_file_infos = {}
    
    for index in range(0, 10):
        file_name, conversion_result = convert(file_path=file_path,
                                            data_type=data_types[index],
                                            sparsity=sparsity[index],
                                            use_dla=use_dla[index],
                                            batch_size=batch_size, channel_size=channel_size,input_width=input_width, input_height=input_height,
                                            output_folder=output_folder)

        converted_file_infos[conversion_result["trt_file_name"]] = conversion_result
    json_file_path = os.path.join(output_folder, f"{file_name}.json")
    save_dict_to_json(converted_file_infos, json_file_path)
    return file_name, converted_file_infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX file to 10 TensorRT files with different options")
    parser.add_argument('--onnx', type=str, help="The path to the onnx file")
    parser.add_argument('--input_width', type=int, help="The input width for the model")
    parser.add_argument('--input_height', type=int, help="The input height for the model")
    parser.add_argument('--output_folder', type=str, default="./output", help="The path to the onnx file")
    
    args = parser.parse_args()
    file_name, converted_file_infos = convert_all(file_path=args.onnx, input_width=args.input_width, input_height=args.input_height, output_folder=args.output_folder)
    json_file_path = os.path.join(args.output_folder, f"{file_name}.json")
    save_dict_to_json(converted_file_infos, json_file_path)
    print("TensorRT conversion is Done!!")
    print(f"JSON Data is generated :{json_file_path}")
    
    