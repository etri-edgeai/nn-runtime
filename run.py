import argparse
import copy
import cv2
import numpy as np
import os
import glob
import tflite_runtime.interpreter as tflite
import time
from PIL import Image, ImageOps
from utils import scale_coords

class Detector:

    def __init__(
            self,
            tflite_path = 'yolov5s-fp16.tflite', 
            conf_thres=0.25,
            iou_thres=0.45
            ):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # prepare model
        self.interpreter = tflite.Interpreter(tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"][1:]
        self.img_size = self.input_shape[:-1]
        self.input_data = np.ndarray(shape=(1, *self.input_shape), dtype=np.float32)

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    

    def non_max_suppression(self,boxes, scores, threshold):	
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
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                            areas[scores_indexes])
            filtered_indexes = set((ious > threshold).nonzero()[0])
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)


    def compute_iou(self,box, boxes, box_area, boxes_area):
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

    def nms(self, prediction):
        prediction = prediction[prediction[...,4] > self.conf_thres]
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        boxes = self.xywh2xyxy(prediction[:, :4])
        
        res = self.non_max_suppression(boxes,prediction[:,4],self.iou_thres)
        
        result_boxes = []
        result_scores = []
        for r in res:
            result_boxes.append(boxes[r])
            result_scores.append(prediction[r,4])
        return result_boxes, result_scores

    def detect(self, original_size, input_image):
        self.input_data[0] = input_image

        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Denormalize xywh
        pred[..., 0] *= original_size[1]  # x
        pred[..., 1] *= original_size[0]  # y
        pred[..., 2] *= original_size[1]  # w
        pred[..., 3] *= original_size[0]  # h

        result_boxes, result_scores = self.nms(pred)

        return result_boxes, result_scores

def run_on_image(detector, image_path):

    image = Image.open(image_path)
    plot_img = copy.deepcopy(image)
    plot_img = image.resize((480, 480))
    plot_imgsize = plot_img.size[:2]
    plot_img = np.asarray(plot_img)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    
    image = image.resize(detector.img_size)

    input_img = np.asarray(image)
    input_img = input_img.astype(np.float16) / 255.0
    
    start_time = time.time()
    result_boxes, result_scores = detector.detect(plot_imgsize, input_img)
    end_time = time.time()

    if len(result_boxes) > 0:
        # result_boxes = scale_coords(detector.img_size, np.array(result_boxes),(original_imgsize[1], original_imgsize[0]))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org 
        org = (20, 40) 
            
        # fontScale 
        fontScale = 0.5
            
        # Blue color in BGR 
        color = (0, 255, 0) 
            
        # Line thickness of 1 px 
        thickness = 1

        for i,r in enumerate(result_boxes):
            org = (int(r[0]),int(r[1]))
            cv2.rectangle(plot_img, (int(r[0]),int(r[1])), (int(r[2]),int(r[3])), (255,0,0), 5)
        cv2.putText(
                img=plot_img,
                text=f"Count: {len(result_boxes)}",
                org=(0, 20), 
                fontFace=1,
                fontScale=2,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA)
        cv2.putText(
                img=plot_img,
                text=f"Latency: {(end_time - start_time)*1000:.2f}ms",
                org=(0, 40), 
                fontFace=1,
                fontScale=2,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

        cv2.imshow('inference', plot_img)
        cv2.waitKey(1)
    print(f'Detector only: {(end_time - start_time):.2f}s')
    print(f'Total Time Taken: {(time.time()-start_time):.2f}s')

def run_on_folder(detector, folder_path):
    for img in sorted(glob.glob(os.path.join(folder_path, '*.jpg'))):
        run_on_image(detector, img)

def run_on_video(detector, video_path):
    video = cv2.VideoCapture(video_path)
    h, w = int(video.get(3)), int(video.get(4))

    no_of_frames = 0
    while True:
        check, frame = video.read()

        if not check:
            break
        no_of_frames += 1

        img = Image.fromarray(frame)
        img = img.resize(detector.img_size)
        input_img = np.asarray(img)
        input_img = input_img.astype(np.float16) / 255.0

        start_time = time.time()
        result_boxes, result_scores = detector.detect((h, w), input_img)
        end_time = time.time()

        for i,r in enumerate(result_boxes):
            org = (int(r[0]),int(r[1]))
            cv2.rectangle(frame, (int(r[0]),int(r[1])), (int(r[2]),int(r[3])), (255,0,0), 10)
        cv2.putText(
                img=frame,
                text=f"Count: {len(result_boxes)}",
                org=(0, 100), 
                fontFace=1,
                fontScale=10,
                color=(0, 255, 0),
                thickness=10,
                lineType=cv2.LINE_AA)
        cv2.putText(
                img=frame,
                text=f"FPS: {1/(end_time - start_time):.2f}",
                org=(0, 200), 
                fontFace=1,
                fontScale=10,
                color=(0, 0, 255),
                thickness=10,
                lineType=cv2.LINE_AA)

        frame = cv2.resize(frame, (480, 480))
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tflite', type=str, default='models/modelsearch3_fast.tflite', help='path to tflite')
    parser.add_argument('-i', '--imgs', type=str, help='path to data')
    parser.add_argument('-v', '--video', type=str, default='Lab1.avi', help='path to video')
    parser.add_argument('--conf_thres', type=float, default=0.3)
    parser.add_argument('--iou_thres', type=float, default=0.45)

    opt = parser.parse_args()

    # create detector
    det = Detector(tflite_path=opt.tflite, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)

    if opt.imgs:
        run_on_folder(det, opt.imgs)
    elif opt.video:
        run_on_video(det, opt.video)
