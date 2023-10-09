"""TFLITE INFERENCE
Todo
"""
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except ImportError:
        raise ImportError("Failed to load tensorflow")
        
import os
import numpy
from ...models.enum import DataAttribute
from typing import Tuple, List, Dict, NewType
from ...models.error import AlreadyInitializedError, UninitializedError

__interpreter_dict = {}
INPUT = 'input'
OUTPUT= 'output'
INTERPRETER = 'interpreter'

input_attribute = NewType('input_attribute', Dict[int, DataAttribute])
output_attribute = NewType('output_attribute', Dict[int, DataAttribute])

def model_initialize(cls:str, num_threads:int=1, **kwargs)->bool:
    """model_initalize는 모델 실행 준비를 하는 함수입니다. 
    모델 실행 전에 반드시 초기화 해야합니다. 
    
    ### tftlie framework

    Parameters
    ----------
    cls: Class, 
        해당 함수를 호출하는 클래스입니다.
    num_threads: int, default=1 
        tflite.Interpreter()의 num_threads 값으로 들어갑니다.

    Raises
    ----------
    AlreadyInitializedError:
        이미 초기화가 된 클래스일 경우 에러를 출력합니다.
    
    """
    if is_init(cls) is True:
        raise AlreadyInitializedError(cls.__name__)

    with open(os.path.join(os.path.dirname(__file__), "{{tflite}}"), "rb") as f: 
        interpreter_obj = tflite.Interpreter(
            model_content=f.read(), 
            num_threads=num_threads
        )
    
    interpreter_obj.allocate_tensors()
    input_attribute, output_attribute = model_input_output_attributes(interpreter_obj)
    
    __interpreter_dict[cls] = {
        INTERPRETER: interpreter_obj,
        INPUT: input_attribute,
        OUTPUT: output_attribute
    }

def model_input_output_attributes(interpreter_object:tflite.Interpreter)->Tuple[input_attribute, output_attribute]:
    """모델의 input, output 정보를 저장합니다.

    Parameters
    ----------
    interpreter_object : tflite.Interpreter
        tflite의 model interpreted object를 매개변수로 받습니다.

    Returns
    -------
    Tuple[input_attribute, output_attribute]
        input의 DataAttribute, output의 DataAttribute를 tuple로 리턴합니다.
    """
    inputs = {}
    outputs = {}
    
    for input_detail in interpreter_object.get_input_details():
        input_data_attribute = DataAttribute()
        input_data_attribute.name = input_detail.get('name')
        input_data_attribute.location = input_detail.get('index')
        input_data_attribute.shape = tuple(input_detail.get('shape'))
        input_data_attribute.dtype = input_detail.get('dtype').__name__
        input_data_attribute.format = "nchw" if input_data_attribute.shape[1] == 3 else "nhwc"
        inputs[input_data_attribute.key] = input_data_attribute    
    
    for output_detail in interpreter_object.get_output_details():
        output_data_attribute = DataAttribute()
        output_data_attribute.name = output_detail.get('name')
        output_data_attribute.location = output_detail.get('index')
        output_data_attribute.shape = tuple(output_detail.get('shape'))
        output_data_attribute.dtype = output_detail.get('dtype').__name__
        outputs[output_data_attribute.key] = output_data_attribute
    
    return inputs, outputs

def model_finalize(cls:str)->None:
    """Todo

    Raises
    --------
    UninitializedError
        초기화 되지 않은 class에 model_inference를 요청하면 에러를 출력합니다.

    """
    if is_init(cls) is False:
        raise UninitializedError
    __interpreter_dict.pop(cls)

def model_inference(cls:str, preprocess_result: Dict[int, numpy.ndarray], **kwargs)\
    ->Dict[int, numpy.ndarray]:
    """Todo

    Raises
    --------
    UninitializedError
        초기화 되지 않은 class에 model_inference를 요청하면 에러를 출력합니다.

    AssertionError
        Todo
    
    InvalidPreprocessDataError
        Todo

    Returns
    --------
    output_dict: Dict[int, numpy.ndarray]
        Todo
    """

    interpreter_dict = __interpreter_dict.get(cls, None)
    if interpreter_dict is None:
        raise UninitializedError
    interpreter_obj = interpreter_dict.get(INTERPRETER)
    output:Dict[int, DataAttribute] = interpreter_dict.get(OUTPUT)

    for location, value in iter(preprocess_result.items()):
        interpreter_obj.set_tensor(location, value)
    interpreter_obj.invoke()

    output_dict = {}
    for output_location in iter(output):
        output_dict[output_location] = interpreter_obj.get_tensor(output_location)
    
    return output_dict

def is_init(cls:str)->bool:
    """입력 받은 클래스가 초기화 되었는지 확인합니다.
    
    Arguments
    --------
    cls: Class
        초기화 상태를 확인하기 위한 클래스

    Returns
    -------
    bool:
        전역변수 __interpreter_dict에 cls이름과 같은 key가 있다면 True를 출력합니다.

    """
    return True if __interpreter_dict.get(cls, None) is not None else False
    
def get_input_output_attributes(cls:str):
    """Todo"""
    dictionary = __interpreter_dict.get(cls, None)
    if dictionary is None:
        return None, None
    return dictionary.get(INPUT, None), dictionary.get(OUTPUT, None)
