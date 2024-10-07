from typing import Dict
from NPBase import Basemodel
import cv2
import numpy as np
class NPModel(Basemodel):
    
    def __init__(self):
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.conf_thres = 0.25
        self.iou_thres = 0.60

        self.input_layer_name = ["input.1"]
        self.input_layer_location = []
        self.output_layer_name = ["665"]
        self.output_layer_location = []

    def preprocess(self, input_data) -> Dict[int, np.ndarray]:
        image = cv2.imread(input_data)
        preprocessed_data = {}
        
        input_keys = self.input_layer_location or self.input_layer_name
        for key in input_keys:
            input_attribute = self.inputs.get(key)
            input_size = [input_attribute.height, input_attribute.width]
            origin_h, origin_w, origin_c = image.shape
            self.origin_h, self.origin_w = origin_h, origin_w
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Calculate width and height and paddings
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
            image = cv2.resize(image, (tw, th))
            # Pad the short side with (128,128,128)
            image = cv2.copyMakeBorder(
                image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
            )
            image = image.astype(np.float32)
            # Normalize to [0,1]
            image /= 255.0 
            # HWC to NHWC format
            image = np.expand_dims(image, axis=0)
            # Convert the image to row-major order:
            data = np.ascontiguousarray(image)
            if input_attribute.format == 'nchw':
                data = data.transpose(0,3,1,2)
            preprocessed_data[key] = data
        return preprocessed_data
    
    def postprocess(self, inference_results):
        output_keys = self.output_layer_location or self.output_layer_name
        # for key in output_keys:
        #     inference_data = inference_results.get(key)
        return inference_results.get(output_keys[0])

    def nms(self, prediction, conf_thres, iou_thres):
        prediction = prediction[prediction[..., 4] > conf_thres]
        boxes = self.xywh2xyxy(prediction[:, :4])
        res = self.non_max_suppression(boxes, prediction[:, 4], iou_thres)
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
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def non_max_suppression(self, boxes, scores, iou_thres):
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
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
            filtered_indexes = np.where(ious > iou_thres)[0]
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)
    
    def compute_iou(self, box, boxes, box_area, boxes_area):
        assert boxes.shape[0] == boxes_area.shape[0]
        ys1 = np.maximum(box[0], boxes[:, 0])
        xs1 = np.maximum(box[1], boxes[:, 1])
        ys2 = np.minimum(box[2], boxes[:, 2])
        xs2 = np.minimum(box[3], boxes[:, 3])
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        unions = box_area + boxes_area - intersections
        ious = intersections / unions
        return ious
    
    def normalize(self, boxes):
        input_keys = self.input_layer_location or self.input_layer_name
        first_input_layer_attribute = self.inputs.get(input_keys[0])
        input_shape = [first_input_layer_attribute.height, first_input_layer_attribute.width]
        if not boxes:
            return boxes
        np_boxes = np.array(boxes)
        if np.all(np_boxes[:,:4] <= 1.0):
            # tflite
            return boxes
        # normalize result
        for box in boxes:
            # tensorrt
            box[0] /= input_shape[1]
            box[1] /= input_shape[0]
            box[2] /= input_shape[1]
            box[3] /= input_shape[0]
        return boxes
    
    def print_result(self, result_label):
        print("--------------------------------------------------------------")
        if result_label == []:
                print(' - Nothing Detected!')
        else:
            for i, label in enumerate(result_label):
                detected = str(self.classes[int(label[5])])
                conf_score = label[4]
                x1, y1, x2, y2 = label[0]*self.origin_w, label[1]*self.origin_h,label[2]*self.origin_w, label[3]*self.origin_h
                print(' - Object {}'.format(i+1))
                print('     - CLASS : {}'.format(detected))
                print('     - SCORE : {:5.4f}'.format(conf_score))
                print('     - BOXES : {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(x1,y1,x2,y2))
        print("--------------------------------------------------------------\n")