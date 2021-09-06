import cv2
import numpy as np
import time
import torch
import torchvision
import argparse
import yaml
import os
import sys

# import tensorrt related
try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda 
except ImportError:
    print("Failed to load tensorrt, pycuda")
    trt = None
    cuda = None

# import tflite related
try:
    import tflite_runtime.interpreter as tflite
    print("Run tflite using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("Run tflite using tensorflow")
    except ImportError:
        tflite = None
        print("Failed to load tf, tflite_runtime")

class ModelWrapper():
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._inputs = None
        self._outputs = None
        self._input_size = None

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._model_path = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    def load_model(self):
        """Set up model. please specify self.model """
        raise NotImplementedError

    def inference(self, input_images):
        """Run inference."""
        raise NotImplementedError


class TRTWrapper(ModelWrapper):
    def __init__(self, model_path, batch):
        super(TRTWrapper, self).__init__(model_path)
        self._batch = batch
        self._bindings = None

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def bindings(self):
        return self._bindings

    @bindings.setter
    def bindings(self, value):
        self._bindings = value

    def load_model(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model_path, 'rb') as f:
            try:
                engine = runtime.deserialize_cuda_engine(f.read())
                self.model = engine
            except:
                sys.exit("Could not load model")
        
        self.alloc_buf()

    def inference(self, input_images):
        inf_res = []
        stream = cuda.Stream()

        for image in input_images:
            self.inputs[0].cpu = image.ravel()
            with self.model.create_execution_context() as context:
                #async version
                [cuda.memcpy_htod_async(inp.gpu, inp.cpu, stream) for inp in self.inputs]
                context.execute_async(self.batch, self.bindings, stream.handle, None)
                [cuda.memcpy_dtoh_async(out.cpu, out.gpu, stream) for out in self.outputs]
                stream.synchronize()

            result = self.outputs[3].cpu
            result = result.reshape((-1, len(classes)+5))
            result = np.expand_dims(result, axis=0)
            result = torch.tensor(result)
            inf_res.append(result)

        return inf_res

    def alloc_buf(self):
        inputs = []
        outputs = []
        bindings = []
        engine = self.model
        input_size = trt.volume(engine.get_binding_shape(0))/3
        input_size = int(input_size**(1/2))

        class HostDeviceMem(object):
            def __init__(self, cpu_mem, gpu_mem):
                self.cpu = cpu_mem
                self.gpu = gpu_mem

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            cpu_mem = cuda.pagelocked_empty(size, dtype)
            gpu_mem = cuda.mem_alloc(cpu_mem.nbytes)
            bindings.append(int(gpu_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(cpu_mem, gpu_mem))
            else:
                outputs.append(HostDeviceMem(cpu_mem, gpu_mem))

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.input_size = input_size


class TFLWrapper(ModelWrapper):
    def __init__(self, model_path):
        super(TFLWrapper, self).__init__(model_path)

    def load_model(self):
        try:
            interpreter = tflite.Interpreter(model_path=self.model_path)
            self.model = interpreter
            self.alloc_buf()
        except ValueError:
            sys.exit("Could not load model")

    def alloc_buf(self):
        interpreter = self.model
        interpreter.allocate_tensors()
        self.inputs = interpreter.get_input_details()
        self.outputs = interpreter.get_output_details()
        self.input_size = self.inputs[0]['shape'][2]
        
    def inference(self, input_images):
        inf_res = []
        interpreter = self.model

        for image in input_images:
            if self.inputs[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = self.inputs[0]["quantization"]
                image = image / input_scale + input_zero_point
            image = image.astype(self.inputs[0]['dtype'])
            interpreter.set_tensor(self.inputs[0]['index'], image)
            interpreter.invoke()
            result = interpreter.get_tensor(self.outputs[3]['index'])
            result = result.reshape((-1, len(classes)+5))
            result = np.expand_dims(result, axis=0)
            result = torch.tensor(result)
            inf_res.append(result)

        return inf_res
        

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    Note:
         Near future, we are considering embeds NMS into ONNX.
         reference: https://github.com/NVIDIA-AI-IOT/yolov4_deepstream/blob/c4a6c2adc862afbae63e44855699ac41ebd9e6c5/tensorrt_yolov4/source/onnx_add_nms_plugin.py#L23
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def preprocess_image(origin_image):
    input_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32)
    input_image /= 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, 0)
        #print('input shape : ' + str(input_image.shape))
    return input_image

def print_result(result):
    for i in range(1, len(res)):
        result = np.array(res[i]).squeeze(axis=0)
        result_image = origin_images[i].copy()
        print("--------------- RESULT ---------------")
        for j in range(result.shape[0]):
            detected = str(classes[int(result[j][5])]).replace('‘', '').replace('’', '')

            confidence_str = str(result[j][4])
            result_image = cv2.rectangle(result_image, (result[j][0], result[j][1]), (result[j][2], result[j][3]), (0, 0, 255), 1)
            result_image = cv2.putText(result_image, str(detected), (int(result[j][0]), int(result[j][1]-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            print("Detect " + str(j) + "(" + str(detected) + ")")
            print("Coordinates : [" + str(result[j][0]) + ", " + str(result[j][1]) + ", " + str(result[j][2]) + ", " + str(result[j][3]) + "]")
            print("Confidence : " + str(result[j][4]))
            print("")
        print("\n\n")
        cv2.imshow("result"+str(i), result_image)
    cv2.waitKey()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--image_folder', required=True, help='image path')
    parser.add_argument('--conf_thres', required=False, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', required=False, default=0.60, help='iou threshold')
    parser.add_argument('--batch', required=False, default=1, help='batch size')
    parser.add_argument('--classes', required=True, help='yaml file with class info')
    args = parser.parse_args()
    
    # load class info(.yaml)
    with open(args.classes) as f:
        classes = yaml.safe_load(f)
        classes = classes['names']

    # load model 
    extension = os.path.splitext(args.model)[1]
    assert extension in ['.trt', '.tflite'], f"Unsupported extension: {extension}, currently supports only either trt or tflite."
    if extension == '.trt':
        assert trt and cuda, f"TensorRT, Pycuda lib loading failed."
        model_wrapper = TRTWrapper(args.model, args.batch)
    elif extension == '.tflite':
        assert tflite, f"Loading lib for running tflite failed, either tensorflow or tflite_runtime is required."
        model_wrapper = TFLWrapper(args.model)

    model_wrapper.load_model()
    input_size = model_wrapper.input_size

    # load and preprocess image
    origin_images = []
    input_images = []
    for filename in os.listdir(args.image_folder):
        img = cv2.imread(os.path.join(args.image_folder, filename))
        img = cv2.resize(img, dsize=(input_size, input_size))
        if img is not None:
            origin_images.append(img)
            input_images.append(preprocess_image(img))

    # inference
    inf_res = model_wrapper.inference(input_images)
    inf_res = torch.stack(inf_res)
    
    # postprocess result(nms)
    res = []
    for i in range(len(inf_res)):
        tmp = inf_res[i]
        tmp = non_max_suppression(prediction=tmp, conf_thres=float(args.conf_thres), iou_thres=float(args.iou_thres), classes=None, agnostic=True)
        tmp = torch.stack(tmp)
        res.append(tmp)

    # print result
    print_result(res)
