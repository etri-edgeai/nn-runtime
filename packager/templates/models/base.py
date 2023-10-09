import abc
import cv2
import logging
import numpy
import os, psutil
from typing import Union, Any, Dict
from ..libs.{{framework}}.inference import model_inference, model_initialize, \
    model_finalize, is_init, get_input_output_attributes

from .enum import DataAttribute
from .error import unsupported_initialize, unsupported_finalize, UninitializedError
from . import log
logger = logging.getLogger(log.__name__)

engine_map = {
            0: {
                'engine_class': '{{engine_classes[0]}}',
                'engine_file': '{{engine_file_names[0]}}',
            },
            1: {
                'engine_class': '{{engine_classes[1]}}',
                'engine_file': '{{engine_file_names[1]}}',
            },
            2: {
                'engine_class': '{{engine_classes[2]}}',
                'engine_file': '{{engine_file_names[2]}}',
            },
            3: {
                'engine_class': '{{engine_classes[3]}}',
                'engine_file': '{{engine_file_names[3]}}',
            },
        }

class Basemodel(metaclass=abc.ABCMeta):
    """Todo
    """
    def __new__(cls): 
        if is_init(cls.__name__) is False:
            raise UninitializedError
        obj = super().__new__(cls)
        setattr(obj, "initialize", unsupported_initialize)
        setattr(obj, "finalize", unsupported_finalize)
        return obj

    @property
    def inputs(cls)->Union[None, Dict[int, DataAttribute]]:
        input, _ = get_input_output_attributes(cls.__name__)
        return input
    
    @property
    def outputs(cls)->Union[None, Dict[int, DataAttribute]]:
        _, output = get_input_output_attributes(cls.__name__)
        return output
    
    @classmethod 
    def initialize(cls, **kwargs)->None:
        """Todo"""
        model_initialize(cls=cls.__name__, **kwargs)

    @classmethod 
    def finalize(cls, **kwargs)->None:
        """Todo"""
        model_finalize(cls=cls.__name__, **kwargs)
    
    def preprocess(self, input_data: Union[str, Any])->Dict[Union[str, int], numpy.ndarray]:
        """A function that preprocesss input data as defined in the input layer of the model.
        Override this function in the child class. It into input data that fits the model.
        
        Parameters
        ----------
        input_data : Union[str, Any]
            The value of input_data recevies the model path or model raw data.

        Returns
        -------
        Dict[Union[str, int], numpy.ndarray]
            Returns the image raw data dictionary.
            The key of the Dictionary is the name or location of the input layer 
            and value is raw data of image
        """
        if isinstance(input_data,str) is True:
            return {next(iter(self.inputs)):cv2.imread(input_data)}
        else:
            return {next(iter(self.inputs)):input_data}
    
    def postprocess(self, inference_result: numpy.ndarray)->Any:
        """Todo"""
        return inference_result

    def run(self, input_data:Any, **kwargs):
        """Todo"""
        pre_result = self.preprocess(input_data)
        engine_cls = self.__class__
        
        pid = os.getpid()
        total_mem_size = psutil.virtual_memory()[0]
        total_cpu_percent = psutil.cpu_percent()
        total_mem_used = psutil.virtual_memory()[3]
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0]
        current_cpu_percent = python_process.cpu_percent()
        cpu_usage = total_cpu_percent - current_cpu_percent
        memory_usage = total_mem_used - memoryUse
        mem_usage = memory_usage/total_mem_size*100.0
        engine_map_index = 0

        if cpu_usage >=0 and cpu_usage < 75 and mem_usage >= 0 and mem_usage < 75:
            engine_map_index = 0
        elif cpu_usage >=75 and mem_usage >= 0 and mem_usage < 75:
            engine_map_index = 1
        elif cpu_usage >=0 and cpu_usage < 75 and mem_usage >= 75:
            engine_map_index = 2
        elif cpu_usage >=75 and mem_usage >= 75:
            engine_map_index = 3

        engine_cls = engine_map[engine_map_index]['engine_class']
        engine_file = engine_map[engine_map_index]['engine_file']

        if is_init(engine_cls) is False:
            model_initialize(engine_cls, engine_file_name=engine_file)

        inference_result = model_inference(cls=engine_cls, preprocess_result=pre_result)
        post_result = self.postprocess(inference_result)
        return post_result