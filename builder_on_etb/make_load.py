import os
import subprocess, signal
import time
import pycuda.driver as cuda
import tensorrt
from contextlib import contextmanager

from NPModel import NPModel

image_path = "/home/nota/test_one_third.jpeg"  #Image path

class NoLogLogger(tensorrt.Logger):
    def log(self, severity, msg):
        # 아무것도 하지 않음, 즉 로그를 출력하지 않음
        pass

class LoadMakerModel(NPModel):
    def __init__(self):
        super().__init__()
        self.input_layer_name = ["input.1"]
        self.input_layer_location = []
        self.output_layer_name = ["665"]
        self.output_layer_location = []

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
cpu_load_python_path = os.path.join(current_directory, "cpu_load.py")
GPU_cmd = '/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nota/EfficientNet-B0-sim-fp32.trt --threads --iterations=999999 > /dev/null 2>&1'
CPU_cmd = f'python3 {cpu_load_python_path} > /dev/null 2>&1'

duration = 45
gpu_load = 76
gpu_processes = 5
cpu_load_percentage = 55



class HostDeviceMem(object):
    def __init__(self, cpu_mem, gpu_mem):
        self.cpu = cpu_mem
        self.gpu = gpu_mem

engine_path = f"/home/nota/EfficientNet-B0-sim-fp32.trt"
    
def load_gpu(engines):
    TRT_LOGGER = NoLogLogger(tensorrt.Logger.ERROR)
    runtime = tensorrt.Runtime(TRT_LOGGER)
    for index in range(0, gpu_load):
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        # input_attribute, output_attribute = model_input_output_attributes(engine)
        inputs = {}
        outputs = {}
        bindings = []
        for binding in engine:
            size = tensorrt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = tensorrt.nptype(engine.get_binding_dtype(binding))
            cpu_mem = cuda.pagelocked_empty(size, dtype)
            gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
            bindings.append(int(gpu_mem))
            if engine.binding_is_input(binding):
                inputs[binding] = HostDeviceMem(cpu_mem, gpu_mem)
            else:
                outputs[binding] = HostDeviceMem(cpu_mem, gpu_mem)

        engines[index] = {
            "interpreter": context,
            "inputs":inputs,
            "outputs":outputs,
            "bindings":bindings,
            "stream":cuda.Stream()
        }

def cpu_low_gpu_low():
    time.sleep(duration)

def cpu_low_gpu_high():
    engines = {}
    load_gpu(engines)
    processes  = []
    for index in range(0, gpu_load):
        p = subprocess.Popen(GPU_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        processes.append(p)

    time.sleep(duration)

    for process in processes:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    for index in range(0, gpu_load):
        engines.pop(index)

def cpu_high_gpu_low():
    CPU_cmd = f'python3 {cpu_load_python_path} {duration} {cpu_load_percentage} > /dev/null 2>&1 '
    p = subprocess.Popen(CPU_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    time.sleep(duration)
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)

def cpu_high_gpu_high():
    processes  = []
    engines = {}
    load_gpu(engines)
    for index in range(0, gpu_load):
        p = subprocess.Popen(GPU_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        processes.append(p)
    CPU_cmd = f'python3 {cpu_load_python_path} {duration} {cpu_load_percentage} > /dev/null 2>&1 '
    p = subprocess.Popen(CPU_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    processes.append(p)
    
    time.sleep(duration)
    for index in range(0, gpu_load):
        engines.pop(index)

    for process in processes:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

load_functions = [cpu_low_gpu_high,
                  cpu_high_gpu_low,
                  cpu_high_gpu_high,
                  cpu_low_gpu_low,
                  cpu_low_gpu_high,
                  cpu_high_gpu_low,
                  cpu_high_gpu_high,
                  ]

def make_cpu_gpu_load():
    cuda.init()
    device = cuda.Device(0)  # enter your Gpu id here
    ctx = device.make_context()
    subprocess.Popen("pkill -9 -f trtexec", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

    # print("\n1. cpu_low_gpu_high!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_low_gpu_high()
    # print("\n2. cpu_high_gpu_low!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_high_gpu_low()
    # print("\n3. cpu_high_gpu_high!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_high_gpu_high()
    # print("\n4. cpu_low_gpu_low!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_low_gpu_low()
    # print("\n5. cpu_low_gpu_high!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_low_gpu_high()
    # print("\n6. cpu_high_gpu_low!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_high_gpu_low()
    # print("\n7. cpu_high_gpu_high!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cpu_high_gpu_high()
    ctx.pop()  # very important
    # print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!")
