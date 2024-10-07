import os, threading, sys, psutil
from NPModel import NPModel
from NPInference import has_interpreter_context
from make_load import *


model_class_names = {{model_class_names}}

class NormalModel(NPModel):
    input_layer_name = {{input_names}}
    input_layer_location = []
    output_layer_name = {{output_names}}
    output_layer_location = []

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


engine_map = {
    'NormalModel' : "{{fp32_trt_file_path}}",
    'FP16Model' : "{{fp16_trt_file_path}}",
    'FP16SparsityModel' : "{{fp16_sparsity_file_path}}",
    'FP16SparsityDLAModel' : "{{fp16_sparsity_dla_file_path}}",
    'INT8Model' : "{{int8_trt_file_path}}",
    'INT8SparsityModel' : "{{int8_sparsity_file_path}}",
    'INT8SparsityDLAModel' : "{{int8_sparsity_dla_file_path}}",
    'MixModel' : "{{mix_trt_file_path}}",
    'MixSparsityModel': "{{mix_trt_sparsity_file_path}}",
    'MixSparsityDLAModel': "{{mix_trt_sparsity_dla_file_path}}",
}

CPU_GPU_MIN_MAX = {{CPU_GPU_MIN_MAX}}

engines = {}

def initialize_modules():
    global engines
    for class_name in model_class_names:
        existing_engine = engines.get(class_name, None)
        if existing_engine is None:
            engine_class = getattr(sys.modules[__name__], class_name)
            engine_path = engine_map[class_name]
            engine_class.initialize(num_threads=1, engine_path=engine_path)
            engine_object = engine_class()
            if has_interpreter_context(engine_class) is False:
                engine_class.finalize()
                if engine_map.get('NormalModel', None) is None:
                    NormalModel.initialize()
                    engine_object = NormalModel()
                else:
                    engine_object = engines.get(model_class_names[0], None)
                
            engines[class_name] = engine_object
            _, _ = engines[class_name].run(image_path)
            print(f'{class_name} is loaded')

def finalize_modules():
    global engines
    # finalize engine classes
    for class_name in model_class_names:
        engine_class = getattr(sys.modules[__name__], class_name)
        if class_name in engines.keys():
            engine = engines.pop(class_name)
            del engine
            engine_class.finalize()

def inference(image_path):
    global engines
    total_mem_size = psutil.virtual_memory()[0]
    pid = os.getpid()

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
            engine_class_name = model_class_names[condition_index]
            return engines[engine_class_name].run(image_path)


if __name__ == "__main__":
    initialize_modules()
    image_path = "test_image.png"
    output, timing = inference(image_path)
    finalize_modules()
# EOF