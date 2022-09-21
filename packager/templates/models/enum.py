from enum import Enum
from typing import List, Tuple, Union

class EnumInputNodeShapeFormat(Enum):
    '''input node shape format의 열거형 집합입니다.
    포맷은 [n, c, h, w]의 조합으로 구성되어 있습니다.
    
    각 알파벳이 의미하는 값은 다음과 같습니다.
    N: number of images in the batch. 만약 'N'이 없다면 싱글 이미지를 뜻합니다.
    C: number of channels of the image
    H: height of the image
    W: width of the image 

    Raises
    ----------
    KeyError
        멤버에 없는 값을 받을 경우 에러를 출력합니다.

    See Also
    ----------

    tensorrt format: 
    https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html
    '''
    """_summary_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    KeyError
        _description_
    """
    NCHW = 'nchw'
    NCWH = 'ncwh'
    NHWC = 'nhwc'
    NWHC = 'nwhc'
    
    CHW = NCHW
    CWH = NCWH
    HWC = NHWC
    WHC = NWHC
    
    # tensorrt format
    LINEAR = NCHW
    CHW2 = NCHW
    HWC8 = NCHW
    CHW4 = NCHW
    CHW16 = NCHW
    CHW32 = NCHW

    UNKNOWN = 'unknown'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)
            


class EnumNodeRawDataType(Enum):
    '''node raw data type의 열거형 집합입니다.

    Raises
    ----------
    KeyError
        멤버에 없는 값을 받을 경우 에러를 출력합니다.

    See Also
    ----------

    tensorrt data type: 
    https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/DataType.html
    '''
    # numpy data type
    FLOAT16='float16'
    FLOAT32='float32'
    FLOAT64='float64'
    INT8='int8'
    INT16='int16'
    INT32='int32'
    INT64='int64'
    UINT8='uint8'
    UINT16='uint16'
    UINT32='utin32'
    UINT64='uint64'
    #BOOL='bool_'

    # tensorrt data type
    FLOAT=FLOAT32
    HALF=FLOAT16

    UNKNOWN = 'unknown'

    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)

