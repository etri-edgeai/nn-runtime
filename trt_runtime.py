import cv2
import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
import torch
import torchvision
import argparse
import yaml

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def alloc_buf(engine):
    inputs = []
    outputs = []
    bindings = []
    input_size = trt.volume(engine.get_binding_shape(0))/3
    input_size = input_size**(1/2)

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

    return inputs, outputs, bindings, input_size


def inference(engine, inputs, outputs, bindings):
    stream = cuda.Stream()

    with engine.create_execution_context() as context:
        #async version
        [cuda.memcpy_htod_async(inp.gpu, inp.cpu, stream) for inp in inputs]
        stream.synchronize()
        context.execute_async(args.batch, bindings, stream.handle, None)
        stream.synchronize()
        [cuda.memcpy_dtoh_async(out.cpu, out.gpu, stream) for out in outputs]
        stream.synchronize()

        # sync version
        #cuda.memcpy_htod(in_gpu, inputs)
        #context.execute(1, [int(in_gpu), int(out_gpu)])
        #cuda.memcpy_dtoh(out_cpu, out_gpu)

    return outputs[3].cpu


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--image', required=True, help='image path')
    parser.add_argument('--conf_thres', required=False, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', required=False, default=0.45, help='iou threshold')
    parser.add_argument('--batch', required=False, default=1, help='batch size')
    parser.add_argument('--classes', required=True, help='yaml file with class info')
    args = parser.parse_args()

    with open(args.classes) as f:
        classes = yaml.safe_load(f)
        classes = classes['class_names']

    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.model, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    # allocate buffer(cpu, gpu)
    inputs, outputs, bindings, input_size = alloc_buf(engine)

    # preprocessing image
    origin_image = cv2.imread(args.image)
    origin_image = cv2.resize(origin_image, dsize=(int(input_size), int(input_size)))
    input_image = preprocess_image(origin_image)
    inputs[0].cpu = input_image.ravel()
    
    # inference
    inf_res = inference(engine, inputs, outputs, bindings)
    
    # postprocessing inference result
    inf_res = inf_res.reshape((-1, len(classes)+5))
    inf_res = np.expand_dims(inf_res, axis=0)
    inf_res = torch.tensor(inf_res)
    inf_res = non_max_suppression(prediction=inf_res, conf_thres=float(args.conf_thres), iou_thres=float(args.iou_thres), classes=None, agnostic=True)
    inf_res = torch.stack(inf_res)
    
    # print result
    result = np.array(inf_res).squeeze(axis=0)
    result_image = origin_image.copy()
    print("--------------- RESULT ---------------")
    for i in range(result.shape[0]):
        detected = str(classes[int(result[i][5])]).replace('‘', '').replace('’', '')

        confidence_str = str(result[i][4])
        result_image = cv2.rectangle(result_image, (result[i][0], result[i][1]), (result[i][2], result[i][3]), (0, 0, 255), 1)
        #result_image = cv2.putText(result_image, str(result[i][4]), (result[i][0], result[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        result_image = cv2.putText(result_image, str(detected), (int(result[i][0]), int(result[i][1]-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        print("Detect " + str(i) + "(" + str(detected) + ")")
        print("Coordinates : [" + str(result[i][0]) + ", " + str(result[i][1]) + ", " + str(result[i][2]) + ", " + str(result[i][3]) + "]")
        print("Confidence : " + str(result[i][4]))
        print("")

#    cv2.imwrite('result_image.jpg', result_image)

    # show result image
    cv2.imshow("Result", result_image)
    cv2.waitKey()

