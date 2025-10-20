import os
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np

# YOLOv3 Infrared
config_car_yolov3 = 'configs/yolov3_simple.py'
checkpoint_car_inf_yolov3 = 'weights/car_infrared/yolov3_92.1.pth'
model_car_inf_yolov3 = init_detector(config_car_yolov3, checkpoint_car_inf_yolov3, device='cuda:0')

# YOLOv3 Visible
checkpoint_car_vis_yolov3 = 'weights/car_visible/yolov3_816.pth'
model_car_vis_yolov3 = init_detector(config_car_yolov3, checkpoint_car_vis_yolov3, device='cuda:0')

def detection(img, model):
    result = inference_detector(model, img)
    score_thres = 0.5
    
    if len(result) == 2:
        result = list(result)
        for i in range(len(result[0])):
            if result[0][i].size != 0:
                bboxes = np.vstack(result[0][i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                segms = result[1][i]
                filtered_segms = []
                for j, flag in enumerate(inds):
                    if flag:
                        filtered_segms.append(segms[j])
                segms = filtered_segms
                result[0][i] = bboxes
                result[1][i] = segms
        result = tuple(result)
    else:
        for i in range(len(result)):
            if result[i].size != 0:
                bboxes = np.vstack(result[i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                result[i] = bboxes
    
    model.show_result(img, result, out_file='result.jpg')
    return result

def yolov3_inf(img):
    result = detection(img, model_car_inf_yolov3)
    return result

def yolov3_vis(img):
    result = detection(img, model_car_vis_yolov3)
    return result
