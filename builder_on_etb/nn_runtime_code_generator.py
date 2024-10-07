import jinja2, argparse, os
from np_utils import load_json_to_dict, CPU_GPU_MIN_MAX

ModelClassMap = {
    "fp32" : {
        "default":"NormalModel",
        "sparsity": "NormalModel",
        "sparsity_dla": "NormalModel",
    },
    "fp16" : {
        "default": "FP16Model",
        "sparsity": "FP16SparsityModel",
        "sparsity_dla": "FP16SparsityDLAModel",
    },
    "int8" : {
        "default": "INT8Model",
        "sparsity": "INT8SparsityModel",
        "sparsity_dla": "INT8SparsityDLAModel",
    },
    "best" : {
        "default": "MixModel",
        "sparsity": "MixSparsityModel",
        "sparsity_dla": "MixSparsityDLAModel",
    }
}

def get_model_class_name(info_dict):
    data_type_resovled = ModelClassMap[info_dict["data_type"]]
    model_class_name = data_type_resovled["default"]
    if info_dict["sparsity"] is True and info_dict["use_dla"] is True:
        model_class_name = data_type_resovled["sparsity_dla"]
    elif info_dict["sparsity"] is True:
        model_class_name = data_type_resovled["sparsity"]
        
    return model_class_name

def generatePythonFile(info_dict, file_name, output_path):    
    try:
        engine_map = info_dict["profile_result"]
        model_class_names = []
        input_names = []
        output_names = []
        
        for profile in engine_map:
            engine_info = info_dict[profile["engine_key"]]
            engine_class = get_model_class_name(engine_info)
            model_class_names.append(engine_class)
            if len(input_names) == 0:
                input_names = engine_info["input_names"]
            if len(output_names) == 0:
                output_names = engine_info["output_names"]
        
        for key in info_dict.keys():
            if key.endswith("fp32.trt"):
                fp32_trt_file_path = info_dict[key]["file_path"]
            if key.endswith("fp16.trt"):
                fp16_trt_file_path = info_dict[key]["file_path"]
            if key.endswith("fp16_sparsity.trt"):
                fp16_sparsity_file_path = info_dict[key]["file_path"]
            if key.endswith("fp16_sparsity_dla.trt"):
                fp16_sparsity_dla_file_path = info_dict[key]["file_path"]
            if key.endswith("int8.trt"):
                int8_trt_file_path = info_dict[key]["file_path"]
            if key.endswith("int8_sparsity.trt"):
                int8_sparsity_file_path = info_dict[key]["file_path"]
            if key.endswith("int8_sparsity_dla.trt"):
                int8_sparsity_dla_file_path = info_dict[key]["file_path"]
            if key.endswith("mix.trt"):
                mix_trt_file_path = info_dict[key]["file_path"]
            if key.endswith("mix_sparsity.trt"):
                mix_trt_sparsity_file_path = info_dict[key]["file_path"]
            if key.endswith("mix_sparsity_dla.trt"):
                mix_trt_sparsity_dla_file_path = info_dict[key]["file_path"]
        
        file_loader = jinja2.FileSystemLoader('./templates')
        env = jinja2.Environment(loader=file_loader)
        template = env.get_template('nn_runtime_template_tensorrt.py')
        python_file_string = template.render(model_class_names = model_class_names,
                                            input_names = input_names,
                                            output_names = output_names,
                                            fp32_trt_file_path = fp32_trt_file_path,
                                            fp16_trt_file_path = fp16_trt_file_path,
                                            fp16_sparsity_file_path = fp16_sparsity_file_path,
                                            fp16_sparsity_dla_file_path = fp16_sparsity_dla_file_path,
                                            int8_trt_file_path = int8_trt_file_path,
                                            int8_sparsity_file_path = int8_sparsity_file_path,
                                            int8_sparsity_dla_file_path = int8_sparsity_dla_file_path,
                                            mix_trt_file_path = mix_trt_file_path,
                                            mix_trt_sparsity_file_path = mix_trt_sparsity_file_path,
                                            mix_trt_sparsity_dla_file_path = mix_trt_sparsity_dla_file_path,
                                            CPU_GPU_MIN_MAX = CPU_GPU_MIN_MAX
                                            )
        output_file_path = os.path.join(output_path, f"{file_name}.py")
        with open(output_file_path, 'w') as output_file:
            output_file.write(python_file_string)
            print(f"{output_file_path} is generated.")

    except Exception as e:
        print("error: " + str(e))
        return False, str(e)

    return True, 'OK'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NN-Runtime Inferece Code.")
    parser.add_argument('--json', type=str, required=True, help="The path to the json file")
    args = parser.parse_args()
    data_dict = load_json_to_dict(args.json)
    directory_path = os.path.dirname(args.json)
    base_name = os.path.basename(args.json)
    file_name, _ = os.path.splitext(base_name)
    generatePythonFile(data_dict, file_name, directory_path)
    