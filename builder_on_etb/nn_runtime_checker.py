import os, threading, psutil, sys, argparse, csv
from NPModel import NPModel
from NPInference import has_interpreter_context
from make_load import *
from np_utils import print_with_overwrite, csv_to_dict, save_dict_to_json, load_json_to_dict, LoadMaker, CPU_GPU_MIN_MAX

class ProfileModel(NPModel):
    input_layer_name = []
    output_layer_name = []
    input_layer_location = []
    output_layer_location = []

class NormalModel(ProfileModel):
    pass

class FP16Model(NormalModel):
    pass

class FP16SparsityModel(NormalModel):
    pass

class FP16SparsityDLAModel(NormalModel):
    pass

class INT8Model(NormalModel):
    pass

class INT8SparsityModel(NormalModel):
    pass

class INT8SparsityDLAModel(NormalModel):
    pass

class MixModel(NormalModel):
    pass

class MixSparsityModel(NormalModel):
    pass

class MixSparsityDLAModel(NormalModel):
    pass


ModelClassMap = {
    "fp32" : {
        "default": NormalModel,
        "sparsity": NormalModel,
        "sparsity_dla": NormalModel,
    },
    "fp16" : {
        "default": FP16Model,
        "sparsity": FP16SparsityModel,
        "sparsity_dla": FP16SparsityDLAModel,
    },
    "int8" : {
        "default": INT8Model,
        "sparsity": INT8SparsityModel,
        "sparsity_dla": INT8SparsityDLAModel,
    },
    "best" : {
        "default": MixModel,
        "sparsity": MixSparsityModel,
        "sparsity_dla": MixSparsityDLAModel,
    }
}

def get_model_class(info_dict):
    data_type_resovled = ModelClassMap[info_dict["data_type"]]
    model_class = data_type_resovled["default"]
    if info_dict["sparsity"] is True and info_dict["use_dla"] is True:
        model_class = data_type_resovled["sparsity_dla"]
    elif info_dict["sparsity"] is True:
        model_class = data_type_resovled["sparsity"]
    
    return model_class

def run_nn_runtime(load_maker: LoadMaker, profile_info, output_path = None, file_name = None):
    ##########################
    ### USE EXISTING VALUE ###
    # csv_file_path = os.path.join(output_path, f"{file_name}_nn_runtime.csv")
    # with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    #     reader = csv.reader(csvfile)
    #     total_lines = sum(1 for row in reader)
    #     return total_lines
    
    #####################
    ###    PROFILE    ###
    engines = {}
    
    engine_map = profile_info["profile_result"]
    image_path = "/home/nota/test_one_third.jpeg"  #Image path
    engine_objects = []
    for profile in engine_map:
        engine_info = profile_info[profile["engine_key"]]
        engine_class = get_model_class(engine_info)
        existing_engine = engines.get(engine_class, None)
        if existing_engine is not None:
            engine_objects.append(existing_engine)
            continue
        
        engine_path = engine_info["file_path"]
        engine_class.initialize(num_threads=1, engine_path=engine_path)
        engine_object = engine_class()
        if has_interpreter_context(engine_class) is False:
            engine_class.finalize()
            if engines.get(FP16Model, None) is None:
                engine_info = profile_info[engine_map[0]["engine_key"]]
                FP16Model.initialize(num_threads=1, engine_path=engine_info["file_path"])
                engine_object = FP16Model()
            else:
                engine_info = profile_info[engine_map[0]["engine_key"]]
                engine_class = get_model_class(engine_info)
                engine_object = engines.get(engine_class, None)
        engine_objects.append(engine_object)
        engine_object.input_layer_name = engine_info["input_names"]
        engine_object.output_layer_name = engine_info["output_names"]
        if engine_object is not None:
            engines[engine_class] = engine_object
            _, _ = engine_object.run(image_path)
            print(f'{engine_object.__class__.__name__} is loaded.')
        else:
            print(f'{engine_object.__class__.__name__} is not loaded!!!! FAIL!! ERROR!!')

    import time
    total_mem_size = psutil.virtual_memory()[0]
    pid = os.getpid()
    csv_header = f'inference_count, cpu_usage - %, memory_usage - MB, memory_usage - %, inference_time - ms'

    csv_data = [csv_header]
    inference_count = 0
    
    load_maker.run()
    while load_maker.is_alive():
        start = time.time()
        end = time.time()
        total_cpu_percent = psutil.cpu_percent()
        total_mem_used = psutil.virtual_memory()[3]
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0]
        current_cpu_percent = python_process.cpu_percent()
        cpu_usage = total_cpu_percent - current_cpu_percent
        memory_usage = total_mem_used - memoryUse
        mem_usage = memory_usage/total_mem_size*100.0
        engine_class_name = 'None'
        for condition_index, condition in enumerate(CPU_GPU_MIN_MAX):
            if (cpu_usage > condition['cpu_min'] and cpu_usage <= condition['cpu_max']) and (mem_usage > condition['gpu_min'] and mem_usage <= condition['gpu_max']):
                engine = engine_objects[condition_index]
                engine_class_name = engine.__class__.__name__
                # print(engine_class_name)
                output, timing = engine_objects[condition_index].run(image_path)
                break

        end = time.time()
        inference_count += 1
        inference_time = (end - start)*1000.0
        csv_row = f'{inference_count}, {cpu_usage}, {memory_usage/1024/1024}, {memory_usage/total_mem_size*100.0}, {inference_time*1000}'
        csv_data.append(csv_row)
        print_with_overwrite(f'{inference_count}: {engine_class_name}: {(end - start)*1000.0} ms, CPU: {cpu_usage}, MEM: {memory_usage/total_mem_size*100.0}            ')

    csv_string = '\n'.join(csv_data)
    if output_path is not None and file_name is not None:
        csv_file_path = os.path.join(output_path, f"{file_name}_nn_runtime.csv")
        with open(csv_file_path, "w") as file:
            file.write(csv_string)
    
    for engine_class in list(engines.keys()).copy():
        engine = engines.pop(engine_class)
        del engine
        engine_class.finalize()
    
    return len(csv_data)
        

def check_nn_runtime(info_dict, json_path):
    load_maker = LoadMaker()
    total_inference_count = run_nn_runtime(load_maker, info_dict)
    test_duration = load_maker.test_duration_in_second()
    original_info_key = None
    for key in info_dict:
        if key.endswith("fp32.trt"):
            original_info_key = key
            
    original_total_inference_count = info_dict[original_info_key]["total_inference_count"]
    inference_latency_enhancement = (test_duration/original_total_inference_count)/(test_duration/total_inference_count)*100
    info_dict["infernece_latency_enhancement"] = inference_latency_enhancement
    save_dict_to_json(info_dict, json_path)
    print("\n\n")
    print("Report Profiling Result:")
    print(f"\tOriginal Model path: \t{info_dict[original_info_key]['file_path']}")
    print(f"\tOriginal Model inference latency: \t{test_duration/original_total_inference_count*1000} ms")
    print(f"\tNN-Runtime inference latency: \t{test_duration/total_inference_count*1000} ms")
    print(f"\tInference latency enhancement: \t{inference_latency_enhancement}%")
    print("Checking NN-Runtime Result Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile mixed TensorRT engines given json information.")
    parser.add_argument('--json', type=str, required=True, help="The path to the json file")
    args = parser.parse_args()
    data_dict = load_json_to_dict(args.json)
    directory_path = os.path.dirname(args.json)
    base_name = os.path.basename(args.json)
    file_name, _ = os.path.splitext(base_name)
    load_maker = LoadMaker()
    test_duration = load_maker.test_duration_in_second()
    total_inference_count = run_nn_runtime(load_maker, data_dict, directory_path, file_name)
    original_info_key = f"{file_name}_fp32.trt"
    original_total_inference_count = data_dict[original_info_key]["total_inference_count"]
    inference_latency_enhancement = (test_duration/original_total_inference_count)/(test_duration/total_inference_count)*100
    data_dict["infernece_latency_enhancement"] = inference_latency_enhancement
    save_dict_to_json(data_dict, args.json)
    print("\n\n")
    print("Report Profiling Result:")
    print(f"\tOriginal Model path: \t{data_dict[original_info_key]['file_path']}")
    print(f"\tOriginal Model inference latency: \t{test_duration/original_total_inference_count*1000} ms")
    print(f"\tNN-Runtime inference latency: \t{test_duration/total_inference_count*1000} ms")
    print(f"\tInference latency enhancement: \t{inference_latency_enhancement}%")
    print("Checking NN-Runtime Result Done.")
# EOF
