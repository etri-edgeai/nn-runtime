import os, threading, psutil, gc, argparse, time, sys
from NPModel import NPModel
from np_utils import csv_to_dict, save_dict_to_json, load_json_to_dict, LoadMaker, CPU_GPU_MIN_MAX

def print_with_overwrite(text):
    sys.stdout.write('\r')
    sys.stdout.write(text + "\t\t")
    sys.stdout.flush() 

class ProfileModel(NPModel):
    input_layer_name = ["input.1"]
    output_layer_name = ["474","477","480","483"]
    input_layer_location = []
    output_layer_location = []

def run_profile(engine_info, image_path = "/home/nota/test_one_third.jpeg"):

    engine_file_name = engine_info["trt_file_name"]
    engine_path = engine_info["file_path"]
    directory_path = os.path.dirname(engine_path)
    cvs_file_name = f"{engine_file_name}.csv"
    cvs_file_path = os.path.join(directory_path, cvs_file_name)
    cvs_file_path = os.path.abspath(cvs_file_path)
    ProfileModel.input_layer_name = engine_info["input_names"]
    ProfileModel.output_layer_name = engine_info["output_names"]
    ProfileModel.initialize(num_threads=1, engine_path=engine_path)

    print(f"Start Profile: {engine_file_name}")
    
    engine = ProfileModel()
    engine.input_layer_name = engine_info["input_names"]
    engine.output_layer_name = engine_info["output_names"]
    begin_time = time.time()
    total_mem_size = psutil.virtual_memory()[0]
    pid = os.getpid()
    csv_header = f'inference_count, cpu_usage - %, memory_usage - MB, memory_usage - %, inference_time - ms'
    # print(cvs_header)
    csv_data = [csv_header]
    load_maker = LoadMaker()
    load_maker.run()
    inference_count = 0
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
        output, timing = engine.run(image_path)
        end = time.time()
        inference_count += 1
        print_with_overwrite(f'{inference_count}: {engine_file_name}: {(end - start)*1000.0} ms, CPU: {cpu_usage}, MEM: {memory_usage/total_mem_size*100.0}       ')

        inference_time = (end - start)*1000.0
        csv_row = f'{inference_count}, {cpu_usage}, {memory_usage/1024/1024}, {memory_usage/total_mem_size*100.0}, {inference_time*1000}'
        csv_data.append(csv_row)

    csv_string = '\n'.join(csv_data)
    with open(cvs_file_path, "w") as file:
        file.write(csv_string)
    ProfileModel.finalize()
    return cvs_file_path

def filter_data(json_data, cpu_min, cpu_max, memory_min, memory_max):
    filtered_data = [
        entry for entry in json_data
        if float(entry[" cpu_usage - %"]) >= cpu_min and float(entry[" cpu_usage - %"]) < cpu_max and float(entry[" memory_usage - %"]) >= memory_min and float(entry[" memory_usage - %"]) < memory_max
    ]
    return filtered_data
    
def profile(info_dict, json_path=None):
    result = []
    for threashold in enumerate(CPU_GPU_MIN_MAX):
        result.append({"engine_key": None, "count": -1})

    info_dict["profile_result"] = result
    if json_path is not None:
        save_dict_to_json(info_dict, json_path)
            
    for key in info_dict.keys():
        if key in ["profile_result", "infernece_latency_enhancement"]:
            continue
        
        ##################
        ###   PROFILE  ###
        gc.enable()
        cvs_file_path = run_profile(engine_info=info_dict[key])
        info_dict[key]["cvs_file_path"] = cvs_file_path
        gc.collect()
        ##################
        
        ##########################
        ### USE EXISTING VALUE ###
        # engine_info=info_dict[key]
        # engine_file_name = engine_info["trt_file_name"]
        # engine_path = engine_info["file_path"]
        # directory_path = os.path.dirname(engine_path)
        # cvs_file_name = f"{engine_file_name}.csv"
        # cvs_file_path = os.path.join(directory_path, cvs_file_name)
        # cvs_file_path = os.path.abspath(cvs_file_path)
        ##########################
            
        dict_data = csv_to_dict(cvs_file_path)
        for index, threasholds in enumerate(CPU_GPU_MIN_MAX):
            filterd_list = filter_data(dict_data,
                                        threasholds['cpu_min'], threasholds['cpu_max'],
                                        threasholds['gpu_min'], threasholds['gpu_max'])
            count = len(filterd_list)
            if result[index]["engine_key"] is None or result[index]["count"] < count:
                result[index]["engine_key"] = key
                result[index]["count"] = count
                result[index]["threashold"] = {"cpu_min": threasholds['cpu_min'],
                                               "cpu_max": threasholds['cpu_max'],
                                               "gpu_min": threasholds['gpu_min'],
                                               "gpu_max": threasholds['gpu_max']}
        info_dict[key]["total_inference_count"] = len(dict_data)
        info_dict["profile_result"] = result
        if json_path is not None:
            save_dict_to_json(info_dict, json_path)
    
    return info_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile TensorRT files with given json information.")
    parser.add_argument('--json', type=str, help="The path to the json file")
    args = parser.parse_args()
    data_dict = load_json_to_dict(args.json)
    data_dict = profile(data_dict, args.json)
    save_dict_to_json(data_dict, args.json)
    
# EOF
