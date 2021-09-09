import cv2
import numpy as np
import time
import argparse
import yaml
import os
import sys

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

# import tensorrt related
try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError:
    print("Failed to load tensorrt, pycuda")
    trt = None
    cuda = None

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
        avg_time = 0

        for image in input_images:
            image = image.transpose(0, 3, 1, 2)
            self.inputs[0].cpu = image.ravel()

            with self.model.create_execution_context() as context:
                #async version
                [cuda.memcpy_htod_async(inp.gpu, inp.cpu, stream) for inp in self.inputs]
                start_time = time.time()
                context.execute_async(self.batch, self.bindings, stream.handle, None)
                end_time = time.time() - start_time
                avg_time += end_time
                [cuda.memcpy_dtoh_async(out.cpu, out.gpu, stream) for out in self.outputs]
                stream.synchronize()

            result = self.outputs[0].cpu if self.model.num_bindings == 1 else self.outputs[3].cpu
            result = result.reshape((-1, len(classes)+5))
            inf_res.append(result.copy())

        avg_time /= len(inf_res)
        print(f"{avg_time:.5f} sec per image (inferece)")
        return inf_res

    def alloc_buf(self):
        inputs = []
        outputs = []
        bindings = []
        engine = self.model
        input_size = engine.get_binding_shape(0)[2:4] if engine.get_binding_shape(0)[1]==3 else input_shape[1:3]

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
        self._dims = "nhwc"

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, value):
        self._dims = value

    def load_model(self):
        try:
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.model = interpreter
            self.alloc_buf()
        except ValueError:
            sys.exit("Could not load model")

    def alloc_buf(self):
        interpreter = self.model
        interpreter.allocate_tensors()
        self.inputs = interpreter.get_input_details()
        self.outputs = interpreter.get_output_details()
        input_shape = self.inputs[0]['shape'] # n h w c
        # configure dims
        self.dims = "nchw" if input_shape[1] == 3 else "nhwc"
        self.input_size = input_shape[2:4] if self.dims=="nchw" else input_shape[1:3]

    def inference(self, input_images):
        inf_res = []
        interpreter = self.model
        avg_time = 0

        for image in input_images:
            if self.inputs[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = self.inputs[0]["quantization"]
                image = image / input_scale + input_zero_point
            image = image.astype(self.inputs[0]['dtype'])
            # reshape images depending on dims(nchw, nhwc)
            if self.dims == "nchw":
                image = image.transpose(0, 3, 1, 2)
            interpreter.set_tensor(self.inputs[0]['index'], image)
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time() - start_time
            avg_time += end_time
            
            result = interpreter.get_tensor(self.outputs[0]['index'] if len(self.outputs) == 1 else self.outputs[3]['index'])
            result = result.reshape((-1, len(classes)+5))
            result = np.expand_dims(result, axis=0)
            inf_res.append(result)
            
        avg_time /= len(inf_res)
        print(f"{avg_time:.5f} sec per image (inference)")
        return inf_res
        

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def normalize(input_shape, boxes):
    """Normalize results in case model returns unnormalized results."""
    if not boxes:
        return boxes
    np_boxes = np.array(boxes)
    if np.all(np_boxes[:,:4] <= 1.0):
        return boxes
    # normalize result
    for box in boxes:
        box[0] /= input_shape[0]
        box[1] /= input_shape[1]
        box[2] /= input_shape[0]
        box[3] /= input_shape[1]
    return boxes

def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
        
        filtered_indexes = np.where(ious > threshold)[0]
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def nms(prediction, conf_thres=0.25, iou_thres=0.45):
    """Calculate nms.

    Return: List[float]: List[w_topleft, h_topleft, w_bottomright, h_bottomright, confidence]
    """
    prediction = prediction[prediction[..., 4] > conf_thres]
    boxes = xywh2xyxy(prediction[:, :4])
    res = non_max_suppression(boxes, prediction[:, 4], iou_thres)
    
    result_boxes = []
    for r in res:
        tmp = np.zeros(6)
        j = prediction[r, 5:].argmax()
        tmp[0] = boxes[r][0].item()
        tmp[1] = boxes[r][1].item()
        tmp[2] = boxes[r][2].item()
        tmp[3] = boxes[r][3].item()
        tmp[4] = prediction[r][4].item()
        tmp[5] = j
        result_boxes.append(tmp)

    return result_boxes

def preprocess_image(raw_bgr_image, input_size):
    """
    Description:
        Converting BGR image to RGB,
        Resizing and padding it to target size,
        Normalizing to [0, 1]
        Transforming to NCHW format
    Argument:
        raw_bgr_image: a numpy array from cv2 (BGR) (H, W, C)
    Return:
        preprocessed_image: preprocessed image (1, C, resized_H, resized_W)
        original_image: the original image (H, W, C)
        origin_h: height of the original image
        origin_w: width of the origianl image
    """

    original_image = raw_bgr_image
    origin_h, origin_w, origin_c = original_image.shape
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_size[1] / origin_w
    r_h = input_size[0] / origin_h
    if r_h > r_w:
        tw = input_size[1]
        th = int(r_w *  origin_h)
        tx1 = tx2 = 0
        ty1 = int((input_size[0] - th) / 2)
        ty2 = input_size[0] - th - ty1
    else:
        tw = int(r_h * origin_w)
        th = input_size[0]
        tx1 = int((input_size[1] - tw) / 2)
        tx2 = input_size[1] - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    preprocessed_image = np.ascontiguousarray(image)

    return preprocessed_image

def print_result(result):
    for i in range(1, len(result)):
        res = np.array(result[i])
        result_image = origin_images[i].copy()
        h, w = result_image.shape[:2]
        print("--------------- RESULT ---------------")
        for j in range(len(result[i])):
            detected = str(classes[int(res[j][5])]).replace('‘', '').replace('’', '')
            confidence_str = str(res[j][4])
            # unnormalize depending on the visualizing image size
            x1 = int(res[j][0] * w)
            y1 = int(res[j][1] * h)
            x2 = int(res[j][2] * w)
            y2 = int(res[j][3] * h)
            result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            result_image = cv2.putText(result_image, str(detected), (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            print("Detect " + str(j) + "(" + str(detected) + ")")
            print("Coordinates : [{:.5f}, {:.5f}, {:.5f}, {:.5f}]".format(x1, y1, x2, y2))
            print("Confidence : {:.7f}".format(res[j][4]))
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
        classes = classes['class_names']

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
        # image load failed
        if img is None:
            continue
        origin_images.append(img)
        input_images.append(preprocess_image(img, input_size))

    # inference
    inf_res = model_wrapper.inference(input_images)
    
    # postprocess result(nms)
    res = []
    for i in range(len(inf_res)):
        tmp = inf_res[i]
        tmp = nms(prediction=tmp, conf_thres=float(args.conf_thres), iou_thres=float(args.iou_thres))
        tmp = normalize(input_size, tmp)
        res.append(tmp)

    # print result
    print_result(res)
